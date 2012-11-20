#!/usr/bin/perl
# Copyright 2011 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
#
# Test script run as a child process under cgi_test.go

use strict;
use Cwd;

binmode STDOUT;

my $q = MiniCGI->new;
my $params = $q->Vars;

if ($params->{"loc"}) {
    print "Location: $params->{loc}\r\n\r\n";
    exit(0);
}

print "Content-Type: text/html\r\n";
print "X-CGI-Pid: $$\r\n";
print "X-Test-Header: X-Test-Value\r\n";
print "\r\n";

if ($params->{"bigresponse"}) {
    for (1..1024) {
        print "A" x 1024, "\r\n";
    }
    exit 0;
}

print "test=Hello CGI\r\n";

foreach my $k (sort keys %$params) {
    print "param-$k=$params->{$k}\r\n";
}

foreach my $k (sort keys %ENV) {
    my $clean_env = $ENV{$k};
    $clean_env =~ s/[\n\r]//g;
    print "env-$k=$clean_env\r\n";
}

# NOTE: msys perl returns /c/go/src/... not C:\go\....
my $dir = getcwd();
if ($^O eq 'MSWin32' || $^O eq 'msys') {
    if ($dir =~ /^.:/) {
        $dir =~ s!/!\\!g;
    } else {
        my $cmd = $ENV{'COMSPEC'} || 'c:\\windows\\system32\\cmd.exe';
        $cmd =~ s!\\!/!g;
        $dir = `$cmd /c cd`;
        chomp $dir;
    }
}
print "cwd=$dir\r\n";

# A minimal version of CGI.pm, for people without the perl-modules
# package installed.  (CGI.pm used to be part of the Perl core, but
# some distros now bundle perl-base and perl-modules separately...)
package MiniCGI;

sub new {
    my $class = shift;
    return bless {}, $class;
}

sub Vars {
    my $self = shift;
    my $pairs;
    if ($ENV{CONTENT_LENGTH}) {
        $pairs = do { local $/; <STDIN> };
    } else {
        $pairs = $ENV{QUERY_STRING};
    }
    my $vars = {};
    foreach my $kv (split(/&/, $pairs)) {
        my ($k, $v) = split(/=/, $kv, 2);
        $vars->{_urldecode($k)} = _urldecode($v);
    }
    return $vars;
}

sub _urldecode {
    my $v = shift;
    $v =~ tr/+/ /;
    $v =~ s/%([a-fA-F0-9][a-fA-F0-9])/pack("C", hex($1))/eg;
    return $v;
}
