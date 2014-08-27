#!/usr/bin/perl

use strict;

if ($ENV{GOOS} eq "nacl") {
    delete $ENV{GOROOT_FINAL};
    exec("./nacltest.bash", @ARGV);
    die "Failed to run nacltest.bash: $!\n";
}

exec("./all.bash", @ARGV);
die "Failed to run all.bash: $!\n";

