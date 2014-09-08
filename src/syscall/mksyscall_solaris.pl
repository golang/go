#!/usr/bin/env perl
# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# This program reads a file containing function prototypes
# (like syscall_solaris.go) and generates system call bodies.
# The prototypes are marked by lines beginning with "//sys"
# and read like func declarations if //sys is replaced by func, but:
#	* The parameter lists must give a name for each argument.
#	  This includes return parameters.
#	* The parameter lists must give a type for each argument:
#	  the (x, y, z int) shorthand is not allowed.
#	* If the return parameter is an error number, it must be named err.
#	* If go func name needs to be different than its libc name, 
#	* or the function is not in libc, name could be specified
#	* at the end, after "=" sign, like
#	  //sys getsockopt(s int, level int, name int, val uintptr, vallen *_Socklen) (err error) = libsocket.getsockopt

use strict;

my $cmdline = "mksyscall_solaris.pl " . join(' ', @ARGV);
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
	print STDERR "usage: mksyscall_solaris.pl [-b32 | -l32] [file ...]\n";
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
	my $nonblock = /^\/\/sysnb /;
	next if !/^\/\/sys / && !$nonblock;

	my $syscalldot = "";
	$syscalldot = "syscall." if $package ne "syscall";

	# Line must be of the form
	#	func Open(path string, mode int, perm int) (fd int, err error)
	# Split into name, in params, out params.
	if(!/^\/\/sys(nb)? (\w+)\(([^()]*)\)\s*(?:\(([^()]+)\))?\s*(?:=\s*(?:(\w*)\.)?(\w*))?$/) {
		print STDERR "$ARGV:$.: malformed //sys declaration\n";
		$errors = 1;
		next;
	}
	my ($nb, $func, $in, $out, $modname, $sysname) = ($1, $2, $3, $4, $5, $6);

	# Split argument lists on comma.
	my @in = parseparamlist($in);
	my @out = parseparamlist($out);

	# So file name.
	if($modname eq "") {
		$modname = "libc";
	}
	my $modvname = "mod$modname";
	if($modnames !~ /$modname/) {
		$modnames .= ".$modname";
		$mods .= "\t$modvname = ${syscalldot}newLazySO(\"$modname.so\")\n";
	}

	# System call name.
	if($sysname eq "") {
		$sysname = "$func";
	}

	# System call pointer variable name.
	my $sysvarname = "proc$sysname";

	my $strconvfunc = "BytePtrFromString";
	my $strconvtype = "*byte";

	# Library proc address variable.
	$sysname =~ y/A-Z/a-z/; # All libc functions are lowercase.
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
	my @uses = ();
	my $n = 0;
	foreach my $p (@in) {
		my ($name, $type) = parseparam($p);
		if($type =~ /^\*/) {
			push @args, "uintptr(unsafe.Pointer($name))";
		} elsif($type eq "string" && $errvar ne "") {
			$text .= "\tvar _p$n $strconvtype\n";
			$text .= "\t_p$n, $errvar = $strconvfunc($name)\n";
			$text .= "\tif $errvar != nil {\n\t\treturn\n\t}\n";
			push @args, "uintptr(unsafe.Pointer(_p$n))";
			push @uses, "use(unsafe.Pointer(_p$n))";
			$n++;
		} elsif($type eq "string") {
			print STDERR "$ARGV:$.: $func uses string arguments, but has no error return\n";
			$text .= "\tvar _p$n $strconvtype\n";
			$text .= "\t_p$n, _ = $strconvfunc($name)\n";
			push @args, "uintptr(unsafe.Pointer(_p$n))";
			push @uses, "use(unsafe.Pointer(_p$n))";
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
	}
	my $nargs = @args;

	# Determine which form to use; pad args with zeros.
	my $asm = "${syscalldot}sysvicall6";
	if ($nonblock) {
		$asm = "${syscalldot}rawSysvicall6";
	}
	if(@args <= 6) {
		while(@args < 6) {
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
	my $do_errno = 0;
	for(my $i=0; $i<@out; $i++) {
		my $p = $out[$i];
		my ($name, $type) = parseparam($p);
		my $reg = "";
		if($name eq "err") {
			$reg = "e1";
			$ret[2] = $reg;
			$do_errno = 1;
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
		if($reg ne "e1") {
			$body .= "\t$name = $type($reg)\n";
		}
	}
	if ($ret[0] eq "_" && $ret[1] eq "_" && $ret[2] eq "_") {
		$text .= "\t$call\n";
	} else {
		$text .= "\t$ret[0], $ret[1], $ret[2] := $call\n";
	}
	foreach my $use (@uses) {
		$text .= "\t$use\n";
	}
	$text .= $body;

	if ($do_errno) {
		$text .= "\tif e1 != 0 {\n";
		$text .= "\t\terr = e1\n";
		$text .= "\t}\n";
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
