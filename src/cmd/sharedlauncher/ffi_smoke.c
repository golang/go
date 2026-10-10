#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>
#include <string.h>

typedef int (__cdecl *tool_main_fn)(int, char **);

static int invoke(HMODULE module, int argc, char **argv) {
	tool_main_fn tool_main = (tool_main_fn)GetProcAddress(module, "dll_go_main");
	if (tool_main == NULL) {
		fprintf(stderr, "missing dll_go_main: %lu\n", GetLastError());
		return 127;
	}
	return tool_main(argc, argv);
}

int main(int argc, char **argv) {
	if (argc < 4) {
		fprintf(stderr, "usage: ffi_smoke <tool.dll> <entry-point> <argv0> [arguments...]\n");
		return 2;
	}
	HMODULE module = LoadLibraryA(argv[1]);
	if (module == NULL) {
		fprintf(stderr, "cannot load %s: %lu\n", argv[1], GetLastError());
		return 127;
	}

	int tool_argc = argc - 3;
	char **tool_argv = argv + 3;
	int status = invoke(module, tool_argc, tool_argv);
	if (status != 0) return status;
	if (strcmp(argv[2], "dll_go_main") == 0) {
		char *env[] = {"go", "env", "GOROOT"};
		return invoke(module, 3, env);
	}
	return 0;
}
