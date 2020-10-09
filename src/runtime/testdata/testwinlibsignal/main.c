#include <windows.h>
#include <stdio.h>

HANDLE waitForCtrlBreakEvent;

BOOL WINAPI CtrlHandler(DWORD fdwCtrlType)
{
    switch (fdwCtrlType)
    {
    case CTRL_BREAK_EVENT:
        SetEvent(waitForCtrlBreakEvent);
        return TRUE;
    default:
        return FALSE;
    }
}

int main(void)
{
    waitForCtrlBreakEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
    if (!waitForCtrlBreakEvent) {
        fprintf(stderr, "ERROR: Could not create event");
        return 1;
    }

    if (!SetConsoleCtrlHandler(CtrlHandler, TRUE))
    {
        fprintf(stderr, "ERROR: Could not set control handler");
        return 1;
    }

    // The library must be loaded after the SetConsoleCtrlHandler call
    // so that the library handler registers after the main program.
    // This way the library handler gets called first.
    HMODULE dummyDll = LoadLibrary("dummy.dll");
    if (!dummyDll) {
        fprintf(stderr, "ERROR: Could not load dummy.dll");
        return 1;
    }

    printf("ready\n");
    fflush(stdout);

    if (WaitForSingleObject(waitForCtrlBreakEvent, 5000) != WAIT_OBJECT_0) {
        fprintf(stderr, "FAILURE: No signal received");
        return 1;
    }

    return 0;
}
