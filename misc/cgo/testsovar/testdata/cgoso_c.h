// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

#ifdef WIN32
#if defined(EXPORT_DLL)
#    define VAR __declspec(dllexport)
#elif defined(IMPORT_DLL)
#    define VAR __declspec(dllimport)
#endif
#else
#    define VAR extern
#endif

VAR const char *exported_var;
