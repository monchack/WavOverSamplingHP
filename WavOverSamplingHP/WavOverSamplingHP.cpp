
#include <iostream>
#include <immintrin.h>

#include <Windows.h>

// reomve comment out below to use Boost
#include <boost/multiprecision/cpp_dec_float.hpp>

// Tap size; change this number if necessary. Must be an odd number
//#define TAP_SIZE 4095
//#define TAP_SIZE 524287
#define TAP_SIZE 65535

//#define HIGH_PRECISION 1


#define DATA_UNIT_SIZE (1024 * 512)

// 16(15+1)bit  X  scale: 48(47+1)bit =  63(62+1)bit -> 32bit (31bit shift)
#define COEFF_SCALE 47
#define SCALE_SHIFT 31

using namespace boost::multiprecision;
using boost::math::constants::pi;

void createHannCoeff(int tapNum, long long* dest, double* dest2)
{
	int coeffNum = (tapNum + 1) / 2;
	cpp_dec_float_100* coeff1 = (cpp_dec_float_100*)::GlobalAlloc(GPTR, sizeof(cpp_dec_float_100) * coeffNum);
	cpp_dec_float_100* coeff2 = (cpp_dec_float_100*)::GlobalAlloc(GPTR, sizeof(cpp_dec_float_100) * coeffNum);
	cpp_dec_float_100* coeff3 = (cpp_dec_float_100*)::GlobalAlloc(GPTR, sizeof(cpp_dec_float_100) * coeffNum);

	cpp_dec_float_100 piq = pi<cpp_dec_float_100>();

	coeff1[0] = 2.0f * (22050.0f / 352800.0f);
	for (int i = 1; i < coeffNum; ++i)
	{
		cpp_dec_float_100 x = i * 2 * piq * 22050 / 352800;
		coeff1[i] = boost::multiprecision::sin(x) / (piq * i);
	}

	#pragma omp parallel for
	for (int i = 0; i < coeffNum; ++i)
	{
		cpp_dec_float_100 x = 2.0 * piq * i / (tapNum - 1);
		coeff2[i] = 0.5 + 0.5 * boost::multiprecision::cos(x);
	}
	coeff2[coeffNum - 1] = 0;

	long long scale = 1LL << (COEFF_SCALE + 3);

	#pragma omp parallel for
	for (int i = 0; i < coeffNum; ++i)
	{
		coeff3[i] = coeff1[i] * coeff2[i] * scale;
	}

	dest[coeffNum - 1] = (long long)boost::multiprecision::round(coeff3[0]);
	dest2[coeffNum - 1] = (double)(coeff3[0] - boost::multiprecision::round(coeff3[0]));

	#pragma omp parallel for
	for (int i = 1; i < coeffNum; ++i)
	{
		cpp_dec_float_100 x = boost::multiprecision::round(coeff3[i]);
		dest[coeffNum - 1 + i] = (long long)x;
		dest[coeffNum - 1 - i] = (long long)x;
		dest2[coeffNum - 1 + i] = (double)(coeff3[i] - x);
		dest2[coeffNum - 1 - i] = (double)(coeff3[i] - x);
	}
	::GlobalFree(coeff1);
	::GlobalFree(coeff2);
	::GlobalFree(coeff3);
}

static void writeRaw32bitPCM(long long left, long long right, int* buffer)
{
	int shift = SCALE_SHIFT;

	int add = 1 >> (shift - 1);
	if (left >= 0) left += add;
	else left -= add;
	if (right >= 0) right += add;
	else right -= add;

	if (left >= 4611686018427387904) left = 4611686018427387904 - 1; // over 63bit : limitted to under [1 << 62]   62bit + 1bit
	if (right >= 4611686018427387904) right = 4611686018427387904 - 1;

	if (left < -4611686018427387904) left = -4611686018427387904;
	if (right < -4611686018427387904) right = -4611686018427387904;


	left = left >> shift;
	right = right >> shift;

	buffer[0] = (int)left;
	buffer[1] = (int)right;
}



int do_oversample(short* src, unsigned int length, long long* coeff, double* coeff2, int tapNum, int* dest, int x8pos)
{
	int half_size = (tapNum - 1) / 2;

	long long tmpLeft, tmpRight;

	__m128d tmpLeft2;
	__m128d tmpRight2;
	__m128d x, y;
	__declspec(align(32)) long long tmpLR[4];


	for (unsigned int i = 0; i < length; ++i)
	{
		tmpLeft = 0;
		tmpRight = 0;
		tmpLR[0] = 0;
		tmpLR[1] = 0;
		tmpLeft2 = _mm_setzero_pd();
		tmpRight2 = _mm_setzero_pd();

		x = _mm_setzero_pd();
		y = _mm_setzero_pd();

		short* srcPtr = src + 2;
		long long* coeffPtr = coeff + half_size - x8pos + 8;
		double* coeff2Ptr = coeff2 + half_size - x8pos + 8;

		for (int j = 1; (j * 8 - x8pos) <= half_size; ++j)
		{
			long long srcLeft = (long long)*srcPtr;
			long long srcRight = (long long)*(srcPtr + 1);
			
			tmpLR[0] += srcLeft * *coeffPtr;
			tmpLR[1] += srcRight * *coeffPtr;

			#if defined(HIGH_PRECISION)
			x = _mm_cvtsi64_sd(x, srcLeft); // load "long long" integer (src) and store as double
			y = _mm_cvtsi64_sd(y, srcRight);
			x = _mm_mul_sd(x, _mm_load_sd(coeff2Ptr));
			y = _mm_mul_sd(y, _mm_load_sd(coeff2Ptr));
			tmpLeft2 = _mm_add_sd(tmpLeft2, x);
			tmpRight2 = _mm_add_sd(tmpLeft2, y);
			#endif

			srcPtr += 2;
			coeffPtr += 8;
			coeff2Ptr += 8;
		}

		srcPtr = src;
		coeffPtr = coeff + half_size + x8pos;
		coeff2Ptr = coeff2 + half_size + x8pos;

		for (int j = 0; (j * 8 + x8pos) <= half_size; ++j)
		{
			long long srcLeft = (long long)*srcPtr;
			long long srcRight = (long long)*(srcPtr + 1);

			tmpLR[0] += srcLeft * *coeffPtr;
			tmpLR[1] += srcRight * *coeffPtr;

			#if defined(HIGH_PRECISION)
			x = _mm_cvtsi64_sd(x, srcLeft); // load "long long" integer (src) and store as double
			y = _mm_cvtsi64_sd(y, srcRight);
			x = _mm_mul_sd(x, _mm_load_sd(coeff2Ptr));
			y = _mm_mul_sd(y, _mm_load_sd(coeff2Ptr));
			tmpLeft2 = _mm_add_sd(tmpLeft2, x);
			tmpRight2 = _mm_add_sd(tmpLeft2, y);
			#endif

			srcPtr -= 2;
			coeffPtr += 8;
			coeff2Ptr += 8;
		}

		#if defined(HIGH_PRECISION)
		tmpLR[0] += _mm_cvtsd_si64(tmpLeft2); // get "long long" from double
		tmpLR[1] += _mm_cvtsd_si64(tmpRight2);
		#endif

		writeRaw32bitPCM(tmpLR[0], tmpLR[1], dest + x8pos*2);

		src += 2;
		dest += 8 * 2;
	}
	return 0;
}

int  oversample(short* src, unsigned int length, long long* coeff, double* coeff2, int tapNum, int* dest, unsigned int option)
{
	if (option == 0)
	{
		int half_size = (tapNum - 1) / 2;
		
		for (unsigned int i = 0; i < length; ++i)
		{
			short *srcLeft = src;
			short *srcRight = src + 1;
			long long tmpLeft, tmpRight;

			tmpLeft = *srcLeft * coeff[half_size];
			tmpRight = *srcRight * coeff[half_size];
			writeRaw32bitPCM(tmpLeft, tmpRight, dest);

			src += 2;
			dest += 8 * 2;
		}
	}
	else
	{
		do_oversample(src,  length, coeff, coeff2, tapNum, dest, option);
	}
	return 0;
}


struct oversample_info
{
	short* src;
	unsigned int length;
	long long* coeff;
	double* coeff2;
	int tapNum;
	int* dest;
	int option;
};

DWORD WINAPI ThreadFunc(LPVOID arg)
{
	struct oversample_info* info = (struct oversample_info*)arg;
	oversample(info->src, info->length, info->coeff, info->coeff2, info->tapNum, info->dest, info->option);
	return 0;
}

unsigned int searchFmtDataChunk(wchar_t* fileName, WAVEFORMATEX* wf, DWORD* offset, DWORD* size)
{
	HANDLE fileHandle;
	fileHandle = CreateFileW(fileName, GENERIC_READ, 0, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
	if (fileHandle == INVALID_HANDLE_VALUE)
	{
		return 0;
	}

	DWORD header[2];
	DWORD readSize;
	WORD  wav[8];
	DWORD riffSize, pos = 0;
	DWORD dataOffset, dataSize;
	::ReadFile(fileHandle, header, 8, &readSize, NULL);
	bool fmtFound = false, dataFound = false;

	if (readSize != 8)
	{
		CloseHandle(fileHandle);
		return 0;
	}

	if (header[0] != 0X46464952)
	{
		// not "RIFF"
		CloseHandle(fileHandle);
		return 0;
	}
	riffSize = header[1];

	::ReadFile(fileHandle, header, 4, &readSize, NULL);
	if (readSize != 4)
	{
		CloseHandle(fileHandle);
		return 0;
	}
	if (header[0] != 0x45564157)
	{
		// not "WAVE"
		CloseHandle(fileHandle);
		return 0;
	}
	pos += 4;

	while (pos < riffSize)
	{
		::ReadFile(fileHandle, header, 8, &readSize, NULL);
		if (readSize != 8)
		{
			break;
		}
		pos += 8;

		if (header[0] == 0X20746d66)
		{
			// "fmt "
			if (header[1] >= 16)
			{
				::ReadFile(fileHandle, wav, 16, &readSize, NULL);
				if (readSize != 16)
				{
					break;
				}
				fmtFound = true;
				if (header[1] > 16)
				{
					::SetFilePointer(fileHandle, header[1] - 16, 0, FILE_CURRENT);
				}
				pos += header[1];
			}
			else
			{
				::SetFilePointer(fileHandle, header[1], 0, FILE_CURRENT);
				pos += header[1];
			}
		}
		else if (header[0] == 0X61746164)
		{
			// "data"
			dataFound = true;
			dataOffset = ::SetFilePointer(fileHandle, 0, 0, FILE_CURRENT);
			dataSize = header[1];
			::SetFilePointer(fileHandle, header[1], 0, FILE_CURRENT);
			pos += header[1];
		}
		else
		{
			::SetFilePointer(fileHandle, header[1], 0, FILE_CURRENT);
			pos += header[1];
		}
		if (GetLastError() != NO_ERROR)
		{
			break;
		}
	}
	CloseHandle(fileHandle);

	if (dataFound && fmtFound)
	{
		*offset = dataOffset;
		*size = dataSize;
		wf->wFormatTag = wav[0]; //  1:LPCM   3:IEEE float
		wf->nChannels = wav[1]; //  1:Mono  2:Stereo
		wf->nSamplesPerSec = *(DWORD*)(wav + 2);  // 44100, 48000, 176400, 19200, 352800, 384000...
		wf->nAvgBytesPerSec = *(DWORD*)(wav + 4);
		wf->nBlockAlign = wav[6]; // 4@16bit/2ch,  6@24bit/2ch,   8@32bit/2ch   
		wf->wBitsPerSample = wav[7]; // 16bit, 24bit, 32bit
		wf->cbSize = 0;
		return 1;
	}
	return 0;
}

DWORD readWavFile(wchar_t* fileName, void* readMem, DWORD readPos, DWORD readLength)
{
	HANDLE fileHandle;
	DWORD wavDataOffset, wavDataSize, readSize = 0;
	WAVEFORMATEX wf;

	if (!searchFmtDataChunk(fileName, &wf, &wavDataOffset, &wavDataSize))
	{
		return 0;
	}

	fileHandle = CreateFileW(fileName, GENERIC_READ, 0, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
	if (fileHandle == INVALID_HANDLE_VALUE)
	{
		return 0;
	}

	if (::SetFilePointer(fileHandle, wavDataOffset + readPos, 0, FILE_BEGIN) == INVALID_SET_FILE_POINTER)
	{
		if (GetLastError() != NO_ERROR)
		{
			// fail
			return 0;
		}
	}
	::ReadFile(fileHandle, readMem, readLength, &readSize, NULL);
	::CloseHandle(fileHandle);

	return readSize;
}

static int writePCM352_32_header(HANDLE fileHandle, unsigned long dataSize)
{
	WAVEFORMATEX wf;
	wf.wFormatTag = 0x01;
	wf.nChannels = 2;
	wf.nSamplesPerSec = 352800;
	wf.nAvgBytesPerSec = 352800 * 8; // 352800 * 4byte(32bit) * 2ch
	wf.nBlockAlign = 8; // 8bytes (32bit, 2ch) per sample
	wf.wBitsPerSample = 32;
	wf.cbSize = 0; // ignored. not written.

	DWORD writtenSize = 0;
	WriteFile(fileHandle, "RIFF", 4, &writtenSize, NULL);
	DWORD size = (dataSize + 44) - 8;
	WriteFile(fileHandle, &size, 4, &writtenSize, NULL);
	WriteFile(fileHandle, "WAVE", 4, &writtenSize, NULL);
	WriteFile(fileHandle, "fmt ", 4, &writtenSize, NULL);
	size = 16;
	WriteFile(fileHandle, &size, 4, &writtenSize, NULL);
	WriteFile(fileHandle, &wf, size, &writtenSize, NULL);
	WriteFile(fileHandle, "data", 4, &writtenSize, NULL);
	size = (DWORD)dataSize;
	WriteFile(fileHandle, &size, 4, &writtenSize, NULL);

	return 0;
}

void* getAlignedMemory(void* mem)
{
	long long n = (long long)mem;
	n = n % 16;
	if (n != 0) n = 16 - n;
	return (void*)((unsigned char*)mem + n);
}

int wmain(int argc, wchar_t *argv[], wchar_t *envp[])
{
	DWORD wavDataOffset, wavDataSize, writtenSize, length, readSize = 0;
	WAVEFORMATEX wf;
	wchar_t* fileName;
	wchar_t* destFileName;

	if (argc < 2) return 0;
	fileName = argv[1];
	destFileName = argv[2];

	ULONGLONG startTime = GetTickCount64();
	ULONGLONG elapsedTime, calcStartTime;

	if (!searchFmtDataChunk(fileName, &wf, &wavDataOffset, &wavDataSize))
	{
		return 0;
	}
	int part = wavDataSize / DATA_UNIT_SIZE;
	if ((wavDataSize %  DATA_UNIT_SIZE) != 0) part += 1;

	void* memWorkBuffer = ::GlobalAlloc(GPTR, DATA_UNIT_SIZE * 3 + 1024);
	void* mem1 = getAlignedMemory(memWorkBuffer);
	void* mem2 = (char*)mem1 + DATA_UNIT_SIZE;
	void* mem3 = (char*)mem2 + DATA_UNIT_SIZE;

	void* memOriginalOutBufer = ::GlobalAlloc(GPTR, DATA_UNIT_SIZE * 8 * 2 + 1024);
	void* memOut = getAlignedMemory(memOriginalOutBufer);

	HANDLE fileOut = CreateFileW(destFileName, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS /*CREATE_NEW*/, FILE_ATTRIBUTE_NORMAL, NULL);
	writePCM352_32_header(fileOut, wavDataSize * 8 * 2);

	void* memFirCoeff1 = ::GlobalAlloc(GPTR, sizeof(long long) * TAP_SIZE + 1024);
	long long* firCoeff = (long long* )getAlignedMemory(memFirCoeff1);
	void* memFirCoeff2 = ::GlobalAlloc(GPTR, sizeof(double) * TAP_SIZE + 1024);
	double* firCoeff2 = (double*)getAlignedMemory(memFirCoeff2);

	createHannCoeff(TAP_SIZE, firCoeff, firCoeff2);
	elapsedTime = GetTickCount64() - startTime;
	calcStartTime = GetTickCount64();
	std::cout << "WavOverSampling: Initialization finished:  " << (elapsedTime / 1000) << "." << (elapsedTime % 1000) << " sec  \n";

	float total = 0.0f;

	for (int i = 0; i <= part; ++i)
	{
		::SetThreadExecutionState(ES_SYSTEM_REQUIRED);
		
		length = readSize;
		::CopyMemory(mem1, mem2, DATA_UNIT_SIZE);
		::CopyMemory(mem2, mem3, DATA_UNIT_SIZE);
		::SecureZeroMemory(mem3, DATA_UNIT_SIZE);
		if (i != part)
		{
			readSize = readWavFile(fileName, mem3, DATA_UNIT_SIZE * i, DATA_UNIT_SIZE);
		}
		if (i == 0) continue;
	
		struct oversample_info info[8];
		info[0].src = (short* )mem2;
		info[0].length = length / 4;
		info[0].coeff = firCoeff;
		info[0].coeff2 = firCoeff2;
		info[0].tapNum = TAP_SIZE;
		info[0].dest = (int* )memOut;
		info[0].option = 0;
		
		HANDLE thread[8];
		DWORD threadId[8];
		for (int j = 0; j < 8; ++j)
		{
			info[j] = info[0];
			info[j].option = j;
			thread[j] = CreateThread(NULL, 0, ThreadFunc, (LPVOID)&info[j], 0, &threadId[j]);
		}
		::WaitForMultipleObjects(8, thread, TRUE, INFINITE);
	
		::WriteFile(fileOut, memOut, length * 8 * 2, &writtenSize, NULL);

		total += (DATA_UNIT_SIZE / 4) * 1000;
		elapsedTime = GetTickCount64() - startTime;
		std::cout << "WavOverSampling: Progress  " << (i * 100) / part << "%    " << (total / 44100 / (GetTickCount64() - calcStartTime)) <<  "x    " << (elapsedTime/1000/60) << " min " << (elapsedTime /1000)%60 << " sec  \r";
	}
	elapsedTime = GetTickCount64() - startTime;
	std::cout << "\nWavOverSampling: Completed.   " << (elapsedTime/1000) << "." << (elapsedTime % 1000) <<  " sec  \n";

	::FlushFileBuffers(fileOut);
	::CloseHandle(fileOut);

	::GlobalFree(memWorkBuffer);
	::GlobalFree(memOriginalOutBufer);
	::GlobalFree(memFirCoeff1);
	::GlobalFree(memFirCoeff2);
	return 0;
}
