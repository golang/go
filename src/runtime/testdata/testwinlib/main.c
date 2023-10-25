#include <stdio.h>
#include <windows.h>
#include "testwinlib.h"

int exceptionCount;
int continueCount;
int unhandledCount;

LONG WINAPI customExceptionHandlder(struct _EXCEPTION_POINTERS *ExceptionInfo)
{
    if (ExceptionInfo->ExceptionRecord->ExceptionCode == EXCEPTION_BREAKPOINT)
    {
        exceptionCount++;
        // prepare context to resume execution
        CONTEXT *c = ExceptionInfo->ContextRecord;
#ifdef _AMD64_
        c->Rip = *(DWORD64 *)c->Rsp;
        c->Rsp += 8;
#elif defined(_X86_)
        c->Eip = *(DWORD *)c->Esp;
        c->Esp += 4;
#else
        c->Pc = c->Lr;
#endif
#ifdef _ARM64_
        // TODO: remove when windows/arm64 supports SEH stack unwinding.
        return EXCEPTION_CONTINUE_EXECUTION;
#endif
    }
    return EXCEPTION_CONTINUE_SEARCH;
}
LONG WINAPI customContinueHandlder(struct _EXCEPTION_POINTERS *ExceptionInfo)
{
    if (ExceptionInfo->ExceptionRecord->ExceptionCode == EXCEPTION_BREAKPOINT)
    {
        continueCount++;
    }
    return EXCEPTION_CONTINUE_SEARCH;
}

LONG WINAPI unhandledExceptionHandler(struct _EXCEPTION_POINTERS *ExceptionInfo) {
    if (ExceptionInfo->ExceptionRecord->ExceptionCode == EXCEPTION_BREAKPOINT)
    {
        unhandledCount++;
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    return EXCEPTION_CONTINUE_SEARCH;
}

void throwFromC()
{
    DebugBreak();
}
int main()
{
    // simulate a "lazily" attached debugger, by calling some go code before attaching the exception/continue handler
    Dummy();
    exceptionCount = 0;
    continueCount = 0;
    void *exceptionHandlerHandle = AddVectoredExceptionHandler(0, customExceptionHandlder);
    if (NULL == exceptionHandlerHandle)
    {
        printf("cannot add vectored exception handler\n");
        fflush(stdout);
        return 2;
    }
    void *continueHandlerHandle = AddVectoredContinueHandler(0, customContinueHandlder);
    if (NULL == continueHandlerHandle)
    {
        printf("cannot add vectored continue handler\n");
        fflush(stdout);
        return 2;
    }
    void *prevUnhandledHandler = SetUnhandledExceptionFilter(unhandledExceptionHandler);
    CallMeBack(throwFromC);
    RemoveVectoredContinueHandler(continueHandlerHandle);
    RemoveVectoredExceptionHandler(exceptionHandlerHandle);
    if (prevUnhandledHandler != NULL)
    {
        SetUnhandledExceptionFilter(prevUnhandledHandler);
    }
    printf("exceptionCount: %d\ncontinueCount: %d\nunhandledCount: %d\n", exceptionCount, continueCount, unhandledCount);
    fflush(stdout);
    return 0;
}
