#include <stdio.h>
#include <windows.h>
#include "testwinlib.h"

int exceptionCount;
int continueCount;
LONG WINAPI customExceptionHandlder(struct _EXCEPTION_POINTERS *ExceptionInfo)
{
    if (ExceptionInfo->ExceptionRecord->ExceptionCode == EXCEPTION_BREAKPOINT)
    {
        exceptionCount++;
        // prepare context to resume execution
        CONTEXT *c = ExceptionInfo->ContextRecord;
        c->Rip = *(ULONG_PTR *)c->Rsp;
        c->Rsp += 8;
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    return EXCEPTION_CONTINUE_SEARCH;
}
LONG WINAPI customContinueHandlder(struct _EXCEPTION_POINTERS *ExceptionInfo)
{
    if (ExceptionInfo->ExceptionRecord->ExceptionCode == EXCEPTION_BREAKPOINT)
    {
        continueCount++;
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
    CallMeBack(throwFromC);
    RemoveVectoredContinueHandler(continueHandlerHandle);
    RemoveVectoredExceptionHandler(exceptionHandlerHandle);
    printf("exceptionCount: %d\ncontinueCount: %d\n", exceptionCount, continueCount);
    fflush(stdout);
    return 0;
}
