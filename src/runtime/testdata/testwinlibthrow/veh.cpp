//go:build ignore

#include <windows.h>

extern "C" __declspec(dllexport)
void RaiseExcept(void)
{
    try
    {
        RaiseException(42, 0, 0, 0);
    }
    catch (...)
    {
    }
}

extern "C" __declspec(dllexport)
void RaiseNoExcept(void)
{
    RaiseException(42, 0, 0, 0);
}

static DWORD WINAPI ThreadRaiser(void* Context)
{
    if (Context)
        RaiseExcept();
    else
        RaiseNoExcept();
    return 0;
}

static void ThreadRaiseXxx(int except)
{
    static int dummy;
    HANDLE thread = CreateThread(0, 0, ThreadRaiser, except ? &dummy : 0, 0, 0);
    if (0 != thread)
    {
        WaitForSingleObject(thread, INFINITE);
        CloseHandle(thread);
    }
}

extern "C" __declspec(dllexport)
void ThreadRaiseExcept(void)
{
    ThreadRaiseXxx(1);
}

extern "C" __declspec(dllexport)
void ThreadRaiseNoExcept(void)
{
    ThreadRaiseXxx(0);
}
