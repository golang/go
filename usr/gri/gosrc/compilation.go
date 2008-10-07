// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Compilation

import Platform "platform"
import Utils "utils"
import Globals "globals"
import Object "object"
import Type "type"
import Universe "universe"
import Scanner "scanner"
import AST "ast"
import Parser "parser"
import Importer "import"
import Exporter "export"
import Printer "printer"
import Verifier "verifier"


func ReadImport(comp* Globals.Compilation, filename string, update bool) (data string, ok bool) {
	if filename == "" {
		panic("illegal package file name");
	}

	// see if it just works
	data, ok = Platform.ReadObjectFile(filename);
	if ok {
		return data, ok;
	}

	if filename[0] == '/' {
		// absolute path
		panic(`don't know how to handle absolute import file path "` + filename + `"`);
	}

	// relative path
	// try relative to the $GOROOT/pkg directory
	std_filename := Platform.GOROOT + "/pkg/" + filename;
	data, ok = Platform.ReadObjectFile(std_filename);
	if ok {
		return data, ok;
	}

	if !update {
		return "", false;
	}

	// TODO BIG HACK - fix this!
	// look for a src file
	// see if it just works
	data, ok = Platform.ReadSourceFile(filename);
	if ok {
		comp.env.Compile(comp, filename + Platform.src_file_ext);
		data, ok = ReadImport(comp, filename, false);
		if ok {
			return data, ok;
		}
	}

	return "", false;
}


export func Import(comp *Globals.Compilation, pkg_file string) *Globals.Package {
	data, ok := ReadImport(comp, pkg_file, comp.flags.update_packages);
	var pkg *Globals.Package;
	if ok {
		pkg = Importer.Import(comp, data);
	}
	return pkg;
}


export func Export(comp *Globals.Compilation, pkg_file string) {
	data := Exporter.Export(comp);
	ok := Platform.WriteObjectFile(pkg_file, data);
	if !ok {
		panic("export failed");
	}
}


export func Compile(comp *Globals.Compilation, src_file string) {
	src, ok := Platform.ReadSourceFile(src_file);
	if !ok {
		print("cannot open ", src_file, "\n");
		return;
	}

	if comp.flags.verbosity > 0 {
		print(src_file, "\n");
	}

	scanner := new(Scanner.Scanner);
	scanner.Open(src_file, src);

	var tstream *chan *Scanner.Token;
	if comp.flags.token_chan {
		tstream = new(chan *Scanner.Token, 100);
		go scanner.Server(tstream);
	}

	parser := new(Parser.Parser);
	parser.Open(comp, scanner, tstream);

	parser.ParseProgram();
	if parser.S.nerrors > 0 {
		return;
	}

	if !comp.flags.ast {
		return;
	}

	Verifier.Verify(comp);

	if comp.flags.print_interface {
		Printer.PrintObject(comp, comp.pkg_list[0].obj, false);
	}

	Export(comp, src_file);
}
