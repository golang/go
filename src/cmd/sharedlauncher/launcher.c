/*
 * Small process launcher for a Go tool built with -buildmode=c-shared.
 * The matching library must sit next to this executable and export
 * dll_<executable-name>_main(int argc, char **argv).
 */

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

typedef int (__cdecl *tool_main_fn)(int, char **);

static char *utf8_arg(const wchar_t *arg) {
	int n = WideCharToMultiByte(CP_UTF8, 0, arg, -1, NULL, 0, NULL, NULL);
	if (n == 0) return NULL;
	char *out = (char *)malloc((size_t)n);
	if (out == NULL || WideCharToMultiByte(CP_UTF8, 0, arg, -1, out, n, NULL, NULL) == 0) {
		free(out);
		return NULL;
	}
	return out;
}

int wmain(int argc, wchar_t **wargv) {
	wchar_t executable[32768];
	DWORD n = GetModuleFileNameW(NULL, executable, (DWORD)(sizeof executable / sizeof executable[0]));
	if (n == 0 || n >= sizeof executable / sizeof executable[0]) {
		fwprintf(stderr, L"shared launcher: cannot determine executable path (error %lu)\n", GetLastError());
		return 127;
	}
	wchar_t *name = wcsrchr(executable, L'\\');
	if (name == NULL) name = wcsrchr(executable, L'/');
	name = name == NULL ? executable : name + 1;
	wchar_t *dot = wcsrchr(name, L'.');
	if (dot != NULL) *dot = L'\0';
	for (wchar_t *p = name; *p; ++p) {
		if (!((*p >= L'a' && *p <= L'z') || (*p >= L'A' && *p <= L'Z') ||
		      (*p >= L'0' && *p <= L'9') || *p == L'_')) {
			fwprintf(stderr, L"shared launcher: unsupported tool name\n");
			return 127;
		}
	}

	wchar_t library[32768];
	if (swprintf(library, sizeof library / sizeof library[0], L"%ls.dll", executable) < 0) {
		fwprintf(stderr, L"shared launcher: library path is too long\n");
		return 127;
	}
	HMODULE module = LoadLibraryW(library);
	if (module == NULL) {
		fwprintf(stderr, L"shared launcher: cannot load %ls (error %lu)\n", library, GetLastError());
		return 127;
	}
	char symbol[256] = "dll_";
	char *utf8_name = utf8_arg(name);
	if (utf8_name == NULL || strlen(utf8_name) + sizeof "dll__main" > sizeof symbol) {
		free(utf8_name);
		FreeLibrary(module);
		fprintf(stderr, "shared launcher: invalid tool name\n");
		return 127;
	}
	strcat(symbol, utf8_name);
	strcat(symbol, "_main");
	free(utf8_name);
	tool_main_fn tool_main = (tool_main_fn)GetProcAddress(module, symbol);
	if (tool_main == NULL) {
		fprintf(stderr, "shared launcher: missing entry point %s (error %lu)\n", symbol, GetLastError());
		FreeLibrary(module);
		return 127;
	}

	char **argv = (char **)calloc((size_t)argc + 1, sizeof *argv);
	if (argv == NULL) {
		fprintf(stderr, "shared launcher: out of memory\n");
		FreeLibrary(module);
		return 127;
	}
	for (int i = 0; i < argc; ++i) {
		argv[i] = utf8_arg(wargv[i]);
		if (argv[i] == NULL) {
			fprintf(stderr, "shared launcher: cannot convert argument %d to UTF-8\n", i);
			for (int j = 0; j < i; ++j) free(argv[j]);
			free(argv);
			FreeLibrary(module);
			return 127;
		}
	}
	int result = tool_main(argc, argv);
	for (int i = 0; i < argc; ++i) free(argv[i]);
	free(argv);
	FreeLibrary(module);
	return result;
}

#else
#define _XOPEN_SOURCE 700
#include <dlfcn.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

typedef int (*tool_main_fn)(int, char **);

static int executable_path(const char *arg0, char *path, size_t size) {
#if defined(__linux__)
	ssize_t n = readlink("/proc/self/exe", path, size - 1);
	if (n >= 0 && (size_t)n < size - 1) { path[n] = '\0'; return 0; }
#elif defined(__APPLE__)
	uint32_t n = (uint32_t)size;
	if (_NSGetExecutablePath(path, &n) == 0) {
		char resolved[PATH_MAX];
		if (realpath(path, resolved) != NULL) { strncpy(path, resolved, size); path[size - 1] = '\0'; }
		return 0;
	}
#endif
	if (realpath(arg0, path) != NULL) return 0;
	if (strlen(arg0) >= size) return -1;
	strcpy(path, arg0);
	return 0;
}

int main(int argc, char **argv) {
	char executable[PATH_MAX];
	if (executable_path(argv[0], executable, sizeof executable) != 0) {
		fprintf(stderr, "shared launcher: cannot determine executable path\n");
		return 127;
	}
	char *name = strrchr(executable, '/');
	name = name == NULL ? executable : name + 1;
	char *dot = strrchr(name, '.');
	if (dot != NULL) *dot = '\0';
	for (const char *p = name; *p; ++p) {
		if (!( (*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z') ||
		       (*p >= '0' && *p <= '9') || *p == '_')) {
			fprintf(stderr, "shared launcher: unsupported tool name\n");
			return 127;
		}
	}
	char library[PATH_MAX];
#if defined(__APPLE__)
	const char *extension = ".dylib";
#else
	const char *extension = ".so";
#endif
	if (snprintf(library, sizeof library, "%s%s", executable, extension) >= (int)sizeof library) {
		fprintf(stderr, "shared launcher: library path is too long\n");
		return 127;
	}
	void *module = dlopen(library, RTLD_NOW | RTLD_LOCAL);
	if (module == NULL) {
		fprintf(stderr, "shared launcher: cannot load %s: %s\n", library, dlerror());
		return 127;
	}
	char symbol[256];
	if (snprintf(symbol, sizeof symbol, "dll_%s_main", name) >= (int)sizeof symbol) {
		fprintf(stderr, "shared launcher: invalid tool name\n");
		dlclose(module);
		return 127;
	}
	dlerror();
	tool_main_fn tool_main = (tool_main_fn)dlsym(module, symbol);
	const char *error = dlerror();
	if (error != NULL) {
		fprintf(stderr, "shared launcher: missing entry point %s: %s\n", symbol, error);
		dlclose(module);
		return 127;
	}
	int result = tool_main(argc, argv);
	dlclose(module);
	return result;
}
#endif
