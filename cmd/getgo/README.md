# getgo

A proof-of-concept command-line installer for Go.

This installer is designed to both install Go as well as do the initial configuration
of setting up the right environment variables and paths.

It will install the Go distribution (tools & stdlib) to "/.go" inside your home directory by default.

It will setup "$HOME/go" as your GOPATH.
This is where third party libraries and apps will be installed as well as where you will write your Go code.

If Go is already installed via this installer it will upgrade it to the latest version of Go.

Currently the installer supports Windows, \*nix and macOS on x86 & x64.
It supports Bash and Zsh on all of these platforms as well as powershell & cmd.exe on Windows.

## Usage

Windows Powershell/cmd.exe:

`(New-Object System.Net.WebClient).DownloadFile('https://get.golang.org/installer.exe', 'installer.exe'); Start-Process -Wait -NonewWindow installer.exe; Remove-Item installer.exe`

Shell (Linux/macOS/Windows):

`curl -LO https://get.golang.org/$(uname)/go_installer && chmod +x go_installer && ./go_installer && rm go_installer`

## To Do

* Check if Go is already installed (via a different method) and update it in place or at least notify the user
* Lots of testing. It's only had limited testing so far.
* Add support for additional shells.

## Development instructions

### Testing

There are integration tests in [`main_test.go`](main_test.go). Please add more
tests there.

#### On unix/linux with the Dockerfile

The Dockerfile automatically builds the binary, moves it to
`/usr/local/bin/getgo` and then unsets `$GOPATH` and removes all `$GOPATH` from
`$PATH`.

```bash
$ docker build --rm --force-rm -t getgo .
...
$ docker run --rm -it getgo bash
root@78425260fad0:~# getgo -v
Welcome to the Go installer!
Downloading Go version go1.8.3 to /usr/local/go
This may take a bit of time...
Adding "export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/go/bin" to /root/.bashrc
Downloaded!
Setting up GOPATH
Adding "export GOPATH=/root/go" to /root/.bashrc
Adding "export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/go/bin:/root/go/bin" to /root/.bashrc
GOPATH has been setup!
root@78425260fad0:~# which go
/usr/local/go/bin/go
root@78425260fad0:~# echo $GOPATH
/root/go
root@78425260fad0:~# echo $PATH
/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/go/bin:/root/go/bin
```

## Release instructions

To upload a new release of getgo, run `./make.bash && ./upload.bash`.
