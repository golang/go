#include <stdio.h>
#include <string.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
typedef HMODULE module_t;
typedef int (__cdecl *tool_main_fn)(int, char **);
static module_t load_module(const char *path) { return LoadLibraryA(path); }
static tool_main_fn find_entry(module_t module, const char *symbol) {
	return (tool_main_fn)GetProcAddress(module, symbol);
}
static const char *load_error(void) { return "LoadLibrary/GetProcAddress failed"; }
#else
#include <dlfcn.h>
typedef void *module_t;
typedef int (*tool_main_fn)(int, char **);
static module_t load_module(const char *path) { return dlopen(path, RTLD_NOW | RTLD_LOCAL); }
static tool_main_fn find_entry(module_t module, const char *symbol) {
	return (tool_main_fn)dlsym(module, symbol);
}
static const char *load_error(void) {
	const char *error = dlerror();
	return error == NULL ? "dynamic loader error" : error;
}
#endif

static int invoke(module_t module, const char *symbol, int argc, char **argv) {
	tool_main_fn tool_main = find_entry(module, symbol);
	if (tool_main == NULL) {
		fprintf(stderr, "missing %s: %s\n", symbol, load_error());
		return 127;
	}
	return tool_main(argc, argv);
}

int main(int argc, char **argv) {
	if (argc < 4) {
		fprintf(stderr, "usage: ffi_smoke <tool-library> <entry-point> <argv0> [arguments...]\n");
		return 2;
	}
	module_t module = load_module(argv[1]);
	if (module == NULL) {
		fprintf(stderr, "cannot load %s: %s\n", argv[1], load_error());
		return 127;
	}
	int tool_argc = argc - 3;
	char **tool_argv = argv + 3;
	int status = invoke(module, argv[2], tool_argc, tool_argv);
	if (status != 0) return status;
	if (strcmp(argv[2], "dll_go_main") == 0) {
		char *env[] = {"go", "env", "GOROOT"};
		return invoke(module, argv[2], 3, env);
	}
	return 0;
}
