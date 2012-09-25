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
#	* If the return parameter is an error number, it must be named err.
#	* If go func name needs to be different from it's winapi dll name,
#	  the winapi name could be specified at the end, after "=" sign, like
#	  //sys LoadLibrary(libname string) (handle uint32, err error) = LoadLibraryA
#	* Each function that returns err needs to supply a condition,
#	  that return value of winapi will be tested against to
#	  detect failure. This would set err to windows "last-error",
#	  otherwise it will be nil. The value can be provided
#	  at end of //sys declaration, like
#	  //sys LoadLibrary(libname string) (handle uint32, err error) [failretval==-1] = LoadLibraryA
#	  and is [failretval==0] by default.

use strict;

my $cmdline = "mksyscall_windows.pl " . join(' ', @ARGV);
my $errors = 0;
my $_32bit = "";

binmode STDOUT;

if($ARGV[0] eq "-b32") {
	$_32bit = "big-endian";
	shift;
} elsif($ARGV[0] eq "-l32") {
	$_32bit = "little-endian";
	shift;
}

if($ARGV[0] =~ /^-/) {
	print STDERR "usage: mksyscall_windows.pl [-b32 | -l32] [file ...]\n";
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

my $package = "";
my $text = "";
my $vars = "";
my $mods = "";
my $modnames = "";
while(<>) {
	chomp;
	s/\s+/ /g;
	s/^\s+//;
	s/\s+$//;
	$package = $1 if !$package && /^package (\S+)$/;
	next if !/^\/\/sys /;

	my $syscalldot = "";
	$syscalldot = "syscall." if $package ne "syscall";

	# Line must be of the form
	#	func Open(path string, mode int, perm int) (fd int, err error)
	# Split into name, in params, out params.
	if(!/^\/\/sys (\w+)\(([^()]*)\)\s*(?:\(([^()]+)\))?\s*(?:\[failretval(.*)\])?\s*(?:=\s*(?:(\w*)\.)?(\w*))?$/) {
		print STDERR "$ARGV:$.: malformed //sys declaration\n";
		$errors = 1;
		next;
	}
	my ($func, $in, $out, $failcond, $modname, $sysname) = ($1, $2, $3, $4, $5, $6);

	# Split argument lists on comma.
	my @in = parseparamlist($in);
	my @out = parseparamlist($out);

	# Dll file name.
	if($modname eq "") {
		$modname = "kernel32";
	}
	my $modvname = "mod$modname";
	if($modnames !~ /$modname/) {
		$modnames .= ".$modname";
		$mods .= "\t$modvname = ${syscalldot}NewLazyDLL(\"$modname.dll\")\n";
	}

	# System call name.
	if($sysname eq "") {
		$sysname = "$func";
	}

	# System call pointer variable name.
	my $sysvarname = "proc$sysname";

	# Returned value when failed
	if($failcond eq "") {
		$failcond = "== 0";
	}

	# Decide which version of api is used: ascii or unicode.
	my $strconvfunc = $sysname !~ /W$/ ? "BytePtrFromString" : "UTF16PtrFromString";
	my $strconvtype = $sysname !~ /W$/ ? "*byte" : "*uint16";

	# Winapi proc address variable.
	$vars .= "\t$sysvarname = $modvname.NewProc(\"$sysname\")\n";

	# Go function header.
	$out = join(', ', @out);
	if($out ne "") {
		$out = " ($out)";
	}
	if($text ne "") {
		$text .= "\n"
	}
	$text .= sprintf "func %s(%s)%s {\n", $func, join(', ', @in), $out;

	# Check if err return available
	my $errvar = "";
	foreach my $p (@out) {
		my ($name, $type) = parseparam($p);
		if($type eq "error") {
			$errvar = $name;
			last;
		}
	}

	# Prepare arguments to Syscall.
	my @args = ();
	my $n = 0;
	my @pin= ();
	foreach my $p (@in) {
		my ($name, $type) = parseparam($p);
		if($type =~ /^\*/) {
			push @args, "uintptr(unsafe.Pointer($name))";
		} elsif($type eq "string" && $errvar ne "") {
			$text .= "\tvar _p$n $strconvtype\n";
			$text .= "\t_p$n, $errvar = $strconvfunc($name)\n";
			$text .= "\tif $errvar != nil {\n\t\treturn\n\t}\n";
			push @args, "uintptr(unsafe.Pointer(_p$n))";
			$n++;
		} elsif($type eq "string") {
			print STDERR "$ARGV:$.: $func uses string arguments, but has no error return\n";
			$text .= "\tvar _p$n $strconvtype\n";
			$text .= "\t_p$n, _ = $strconvfunc($name)\n";
			push @args, "uintptr(unsafe.Pointer(_p$n))";
			$n++;
		} elsif($type =~ /^\[\](.*)/) {
			# Convert slice into pointer, length.
			# Have to be careful not to take address of &a[0] if len == 0:
			# pass nil in that case.
			$text .= "\tvar _p$n *$1\n";
			$text .= "\tif len($name) > 0 {\n\t\t_p$n = \&$name\[0]\n\t}\n";
			push @args, "uintptr(unsafe.Pointer(_p$n))", "uintptr(len($name))";
			$n++;
		} elsif($type eq "int64" && $_32bit ne "") {
			if($_32bit eq "big-endian") {
				push @args, "uintptr($name >> 32)", "uintptr($name)";
			} else {
				push @args, "uintptr($name)", "uintptr($name >> 32)";
			}
		} elsif($type eq "bool") {
 			$text .= "\tvar _p$n uint32\n";
			$text .= "\tif $name {\n\t\t_p$n = 1\n\t} else {\n\t\t_p$n = 0\n\t}\n";
			push @args, "uintptr(_p$n)";
			$n++;
		} else {
			push @args, "uintptr($name)";
		}
		push @pin, sprintf "\"%s=\", %s, ", $name, $name;
	}
	my $nargs = @args;

	# Determine which form to use; pad args with zeros.
	my $asm = "${syscalldot}Syscall";
	if(@args <= 3) {
		while(@args < 3) {
			push @args, "0";
		}
	} elsif(@args <= 6) {
		$asm = "${syscalldot}Syscall6";
		while(@args < 6) {
			push @args, "0";
		}
	} elsif(@args <= 9) {
		$asm = "${syscalldot}Syscall9";
		while(@args < 9) {
			push @args, "0";
		}
	} elsif(@args <= 12) {
		$asm = "${syscalldot}Syscall12";
		while(@args < 12) {
			push @args, "0";
		}
	} elsif(@args <= 15) {
		$asm = "${syscalldot}Syscall15";
		while(@args < 15) {
			push @args, "0";
		}
	} else {
		print STDERR "$ARGV:$.: too many arguments to system call\n";
	}

	# Actual call.
	my $args = join(', ', @args);
	my $call = "$asm($sysvarname.Addr(), $nargs, $args)";

	# Assign return values.
	my $body = "";
	my $failexpr = "";
	my @ret = ("_", "_", "_");
	my @pout= ();
	for(my $i=0; $i<@out; $i++) {
		my $p = $out[$i];
		my ($name, $type) = parseparam($p);
		my $reg = "";
		if($name eq "err") {
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
		my $rettype = $type;
		if($type =~ /^\*/) {
			$reg = "unsafe.Pointer($reg)";
			$rettype = "($rettype)";
		}
		if($i == 0) {
			if($type eq "bool") {
				$failexpr = "!$name";
			} elsif($name eq "err") {
				$ret[$i] = "r1";
				$failexpr = "r1 $failcond";
			} else {
				$failexpr = "$name $failcond";
			}
		}
		$failexpr =~ s/(=)([0-9A-Za-z\-+])/$1 $2/;  # gofmt compatible
		if($name eq "err") {
			# Set err to "last error" only if returned value indicate failure
			$body .= "\tif $failexpr {\n";
			$body .= "\t\tif $reg != 0 {\n";
			$body .= "\t\t\t$name = $type($reg)\n";
			$body .= "\t\t} else {\n";
			$body .= "\t\t\t$name = ${syscalldot}EINVAL\n";
			$body .= "\t\t}\n";
			$body .= "\t}\n";
		} elsif($rettype eq "error") {
			# Set $reg to "error" only if returned value indicate failure
			$body .= "\tif $reg != 0 {\n";
			$body .= "\t\t$name = ${syscalldot}Errno($reg)\n";
			$body .= "\t}\n";
		} else {
			$body .= "\t$name = $rettype($reg)\n";
		}
		push @pout, sprintf "\"%s=\", %s, ", $name, $name;
	}
	if ($ret[0] eq "_" && $ret[1] eq "_" && $ret[2] eq "_") {
		$text .= "\t$call\n";
	} else {
		$text .= "\t$ret[0], $ret[1], $ret[2] := $call\n";
	}
	$text .= $body;
	if(0) {
		$text .= sprintf 'print("SYSCALL: %s(", %s") (", %s")\n")%s', $func, join('", ", ', @pin), join('", ", ', @pout), "\n";
	}

	$text .= "\treturn\n";
	$text .= "}\n";
}

if($errors) {
	exit 1;
}

print <<EOF;
// $cmdline
// MACHINE GENERATED BY THE COMMAND ABOVE; DO NOT EDIT

package $package

import "unsafe"
EOF

print "import \"syscall\"\n" if $package ne "syscall";

print <<EOF;

var (
$mods
$vars
)

$text

EOF
exit 0;
