#!/bin/bash
# Copyright 2014 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e

# Download kernel, sets, etc. from ftp.usa.openbsd.org
if ! [ -e install56.iso ]; then
  curl -O ftp://ftp.usa.openbsd.org/pub/OpenBSD/5.6/amd64/install56.iso
fi

# XXX: Download and save bash, curl, and their dependencies too?
# Currently we download them from the network during the install process.

# Create custom site56.tgz set.
mkdir -p etc
cat >install.site <<EOF
#!/bin/sh
env PKG_PATH=ftp://ftp.usa.openbsd.org/pub/OpenBSD/5.6/packages/amd64 pkg_add -iv bash curl git

# See https://code.google.com/p/google-compute-engine/issues/detail?id=77
echo "ignore classless-static-routes;" >> /etc/dhclient.conf
EOF
cat >etc/rc.local <<EOF
(
  set -x
  echo "starting buildlet script"
  netstat -rn
  cat /etc/resolv.conf
  dig metadata.google.internal
  (
    set -e
    export PATH="\$PATH:/usr/local/bin"
    /usr/local/bin/curl -o /buildlet \$(/usr/local/bin/curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/buildlet-binary-url)
    chmod +x /buildlet
    exec /buildlet
  )
  echo "giving up"
  sleep 10
  halt -p
)
EOF
chmod +x install.site
tar -zcvf site56.tgz install.site etc/rc.local

# Hack install CD a bit.
echo 'set tty com0' > boot.conf
dd if=/dev/urandom of=random.seed bs=4096 count=1
cp install56.iso install56-patched.iso
growisofs -M install56-patched.iso -l -R -graft-points \
  /5.6/amd64/site56.tgz=site56.tgz \
  /etc/boot.conf=boot.conf \
  /etc/random.seed=random.seed

# Initialize disk image.
rm -f disk.raw
qemu-img create -f raw disk.raw 10G

# Run the installer to create the disk image.
expect <<EOF
spawn qemu-system-x86_64 -nographic -smp 2 -drive if=virtio,file=disk.raw -cdrom install56-patched.iso -net nic,model=virtio -net user -boot once=d

expect "boot>"
send "\n"

# Need to wait for the kernel to boot.
expect -timeout 600 "\(I\)nstall, \(U\)pgrade, \(A\)utoinstall or \(S\)hell\?"
send "i\n"

expect "Terminal type\?"
send "vt220\n"

expect "System hostname\?"
send "buildlet\n"

expect "Which network interface do you wish to configure\?"
send "vio0\n"

expect "IPv4 address for vio0\?"
send "dhcp\n"

expect "IPv6 address for vio0\?"
send "none\n"

expect "Which network interface do you wish to configure\?"
send "done\n"

expect "Password for root account\?"
send "root\n"

expect "Password for root account\?"
send "root\n"

expect "Start sshd\(8\) by default\?"
send "yes\n"

expect "Start ntpd\(8\) by default\?"
send "no\n"

expect "Do you expect to run the X Window System\?"
send "no\n"

expect "Do you want the X Window System to be started by xdm\(1\)\?"
send "no\n"

expect "Do you want to suspend on lid close\?"
send "no\n"

expect "Change the default console to com0\?"
send "yes\n"

expect "Which speed should com0 use\?"
send "115200\n"

expect "Setup a user\?"
send "gopher\n"

expect "Full name for user gopher\?"
send "Gopher Gopherson\n"

expect "Password for user gopher\?"
send "gopher\n"

expect "Password for user gopher\?"
send "gopher\n"

expect "Since you set up a user, disable sshd\(8\) logins to root\?"
send "yes\n"

expect "What timezone are you in\?"
send "US/Pacific\n"

expect "Which disk is the root disk\?"
send "sd0\n"

expect "Use DUIDs rather than device names in fstab\?"
send "yes\n"

expect "Use \(W\)hole disk or \(E\)dit the MBR\?"
send "whole\n"

expect "Use \(A\)uto layout, \(E\)dit auto layout, or create \(C\)ustom layout\?"
send "custom\n"

expect "> "
send "z\n"

expect "> "
send "a b\n"
expect "offset: "
send "\n"
expect "size: "
send "1G\n"
expect "FS type: "
send "swap\n"

expect "> "
send "a a\n"
expect "offset: "
send "\n"
expect "size: "
send "\n"
expect "FS type: "
send "4.2BSD\n"
expect "mount point: "
send "/\n"

expect "> "
send "w\n"
expect "> "
send "q\n"

expect "Location of sets\?"
send "cd\n"

expect "Which CD-ROM contains the install media\?"
send "cd0\n"

expect "Pathname to the sets\?"
send "5.6/amd64\n"

expect "Set name\(s\)\?"
send "+*\n"

expect "Set name\(s\)\?"
send " -x*\n"

expect "Set name\(s\)\?"
send " -game*\n"

expect "Set name\(s\)\?"
send " -man*\n"

expect "Set name\(s\)\?"
send "done\n"

expect "Directory does not contain SHA256\.sig\. Continue without verification\?"
send "yes\n"

# Need to wait for previous sets to unpack.
expect -timeout 600 "Location of sets\?"
send "done\n"

expect "Ambiguous: choose dependency for git"
send "0\n"

# Need to wait for install.site to install curl, git, et
expect -timeout 600 "CONGRATULATIONS!"

expect "# "
send "halt\n"

expect "Please press any key to reboot.\n"
send "\n"

expect "boot>"
send "\n"

expect -timeout 600 eof
EOF

# Create Compute Engine disk image.
echo "Archiving disk.raw... (this may take a while)"
tar -Szcf openbsd-amd64-gce.tar.gz disk.raw

echo "Done. GCE image is openbsd-amd64-gce.tar.gz."
