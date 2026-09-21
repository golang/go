//go:build ignore

#include <windows.h>

__declspec(dllexport)
void RaiseNoExcept(void)
{
    RaiseException(42, 0, 0, 0);
}

static DWORD WINAPI ThreadRaiser(void* Context)
{
    RaiseNoExcept();
    return 0;
}

__declspec(dllexport)
void ThreadRaiseNoExcept(void)
{
    HANDLE thread = CreateThread(0, 0, ThreadRaiser,  0, 0, 0);
    if (0 != thread)
    {
        WaitForSingleObject(thread, INFINITE);
        CloseHandle(thread);
    }
}
