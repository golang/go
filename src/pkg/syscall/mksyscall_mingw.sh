#!/usr/bin/perl
# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# This program reads a file containing function prototypes
# (like syscall_darwin.go) and generates system call bodies.
# The prototypes are marked by lines beginning with "//sys"
# and read like func declarations if //sys is replaced by func, but:
#	* The parameter lists must give a name for each argument.
#	  This includes return parameters.
#	* The parameter lists must give a type for each argument:
#	  the (x, y, z int) shorthand is not allowed.
#	* If the return parameter is an error number, it must be named errno.
#	* If go func name needs to be different from it's winapi dll name,
#	  the winapi name could be specified at the end, after "=" sign, like
#	  //sys LoadLibrary(libname string) (handle uint32, errno int) = LoadLibraryA
#	* Each function, that returns errno, needs to supply a number,
#	  that return value of winapi will be tested against to
#	  detect failure. This would set errno to windows "last-error",
#	  otherwise it will be 0. The value can be provided
#	  at end of //sys declaration, like
#	  //sys LoadLibrary(libname string) (handle uint32, errno int) [failretval=-1] = LoadLibraryA
#	  and is 0 by default.

$cmdline = "mksyscall_mingw.sh " . join(' ', @ARGV);
$errors = 0;
$_32bit = "";

if($ARGV[0] eq "-b32") {
	$_32bit = "big-endian";
	shift;
} elsif($ARGV[0] eq "-l32") {
	$_32bit = "little-endian";
	shift;
}

if($ARGV[0] =~ /^-/) {
	print STDERR "usage: mksyscall_mingw.sh [-b32 | -l32] [file ...]\n";
	exit 1;
}

sub parseparamlist($) {
	my ($list) = @_;
	$list =~ s/^\s*//;
	$list =~ s/\s*$//;
	if($list eq "") {
		return ();
	}
	return split(/\s*,\s*/, $list);
}

sub parseparam($) {
	my ($p) = @_;
	if($p !~ /^(\S*) (\S*)$/) {
		print STDERR "$ARGV:$.: malformed parameter: $p\n";
		$errors = 1;
		return ("xx", "int");
	}
	return ($1, $2);
}

$text = "";
$vars = "";
while(<>) {
	chomp;
	s/\s+/ /g;
	s/^\s+//;
	s/\s+$//;
	next if !/^\/\/sys /;

	# Line must be of the form
	#	func Open(path string, mode int, perm int) (fd int, errno int)
	# Split into name, in params, out params.
	if(!/^\/\/sys (\w+)\(([^()]*)\)\s*(?:\(([^()]+)\))?\s*(?:\[failretval=(.*)\])?\s*(?:=\s*(\w*))?$/) {
		print STDERR "$ARGV:$.: malformed //sys declaration\n";
		$errors = 1;
		next;
	}
	my ($func, $in, $out, $failretval, $sysname) = ($1, $2, $3, $4, $5);

	# Split argument lists on comma.
	my @in = parseparamlist($in);
	my @out = parseparamlist($out);

	# System call name.
	if($sysname eq "") {
		$sysname = "$func";
	}

	# System call pointer variable name.
	$sysvarname = "proc$sysname";

	# Returned value when failed
	if($failretval eq "") {
		$failretval = "0";
	}

	# Decide which version of api is used: ascii or unicode.
	if($sysname !~ /W$/) {
		$strconvfunc = "StringBytePtr";
	} else {
		$strconvfunc = "StringToUTF16Ptr";
	}

	# Winapi proc address variable.
	$vars .= sprintf "\t%s = getSysProcAddr(modKERNEL32, \"%s\")\n", $sysvarname, $sysname;

	# Go function header.
	$text .= sprintf "func %s(%s) (%s) {\n", $func, join(', ', @in), join(', ', @out);

	# Prepare arguments to Syscall.
	my @args = ();
	my $n = 0;
	foreach my $p (@in) {
		my ($name, $type) = parseparam($p);
		if($type =~ /^\*/) {
			push @args, "uintptr(unsafe.Pointer($name))";
		} elsif($type eq "string") {
			push @args, "uintptr(unsafe.Pointer($strconvfunc($name)))";
		} elsif($type =~ /^\[\](.*)/) {
			# Convert slice into pointer, length.
			# Have to be careful not to take address of &a[0] if len == 0:
			# pass nil in that case.
			$text .= "\tvar _p$n *$1;\n";
			$text .= "\tif len($name) > 0 { _p$n = \&${name}[0]; }\n";
			push @args, "uintptr(unsafe.Pointer(_p$n))", "uintptr(len($name))";
			$n++;
		} elsif($type eq "int64" && $_32bit ne "") {
			if($_32bit eq "big-endian") {
				push @args, "uintptr($name >> 32)", "uintptr($name)";
			} else {
				push @args, "uintptr($name)", "uintptr($name >> 32)";
			}
		} else {
			push @args, "uintptr($name)";
		}
	}

	# Determine which form to use; pad args with zeros.
	my $asm = "Syscall";
	if(@args <= 3) {
		while(@args < 3) {
			push @args, "0";
		}
	} elsif(@args <= 6) {
		$asm = "Syscall6";
		while(@args < 6) {
			push @args, "0";
		}
	} elsif(@args <= 9) {
		$asm = "Syscall9";
		while(@args < 9) {
			push @args, "0";
		}
	} else {
		print STDERR "$ARGV:$.: too many arguments to system call\n";
	}

	# Actual call.
	my $args = join(', ', @args);
	my $call = "$asm($sysvarname, $args)";

	# Assign return values.
	my $body = "";
	my $failexpr = "";
	my @ret = ("_", "_", "_");
	for(my $i=0; $i<@out; $i++) {
		my $p = $out[$i];
		my ($name, $type) = parseparam($p);
		my $reg = "";
		if($name eq "errno") {
			$reg = "e1";
			$ret[2] = $reg;
		} else {
			$reg = sprintf("r%d", $i);
			$ret[$i] = $reg;
		}
		if($type eq "bool") {
			$reg = "$reg != 0";
		}
		if($type eq "int64" && $_32bit ne "") {
			# 64-bit number in r1:r0 or r0:r1.
			if($i+2 > @out) {
				print STDERR "$ARGV:$.: not enough registers for int64 return\n";
			}
			if($_32bit eq "big-endian") {
				$reg = sprintf("int64(r%d)<<32 | int64(r%d)", $i, $i+1);
			} else {
				$reg = sprintf("int64(r%d)<<32 | int64(r%d)", $i+1, $i);
			}
			$ret[$i] = sprintf("r%d", $i);
			$ret[$i+1] = sprintf("r%d", $i+1);
		}
		if($i == 0) {
			if($type eq "bool") {
				$failexpr = "!$name";
			} else {
				$failexpr = "$name == $failretval";
			}
		}
		if($name eq "errno") {
			# Set errno to "last error" only if returned value indicate failure
			$body .= "\tif $failexpr {\n";
			$body .= "\t\t$name = $type($reg);\n";
			$body .= "\t} else {\n";
			$body .= "\t\t$name = 0;\n";
			$body .= "\t}\n";
		} else {
			$body .= "\t$name = $type($reg);\n";
		}
	}
	if ($ret[0] eq "_" && $ret[1] eq "_" && $ret[2] eq "_") {
		$text .= "\t$call;\n";
	} else {
		$text .= "\t$ret[0], $ret[1], $ret[2] := $call;\n";
	}
	$text .= $body;

	$text .= "\treturn;\n";
	$text .= "}\n\n";
}

if($errors) {
	exit 1;
}

print <<EOF;
// $cmdline
// MACHINE GENERATED BY THE COMMAND ABOVE; DO NOT EDIT

package syscall

import "unsafe"

var (
	modKERNEL32 = loadDll("kernel32.dll")
$vars
)

$text

EOF
exit 0;
