# internal/govulncheck package

This package is a literal copy of the cmd/govulncheck/internal/govulncheck
package in the vuln repo (https://go.googlesource.com/vuln).

The `copy.sh` does the copying, after removing all .go files here. To use it:

1. Clone the vuln repo to a directory next to the directory holding this repo
   (tools). After doing that your directory structure should look something like
   ```
   ~/repos/x/tools/gopls/...
   ~/repos/x/vuln/...
   ```

2. cd to this directory.

3. Run `copy.sh`.

4. Re-add build tags for go1.18