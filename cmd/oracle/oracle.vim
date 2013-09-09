" -*- text -*-
"  oracle.vim -- Vim integration for the Go oracle.
"
"  Load with (e.g.)  :source oracle.vim
"  Call with (e.g.)  :GoOracleDescribe
"  while cursor or selection is over syntax of interest.
"  Run :copen to show the quick-fix file.
"
" This is an absolutely rudimentary integration of the Go Oracle into
" Vim's quickfix mechanism and it needs a number of usability
" improvements before it can be practically useful to Vim users.
" Voluntary contributions welcomed!
"
" TODO(adonovan):
" - prompt/save the buffer if modified.
" - reject buffers with no filename.
" - hide all filenames in quickfix buffer.


" Users should customize this to their analysis scope, e.g. main package(s).
let s:scope = "/home/adonovan/go3/got/d.go"

" The path to the Go oracle executable.
let s:go_oracle = "$GOROOT/bin/oracle"

" Enable Vim to recognize GNU-style 'file:line.col-line.col: message' format.
set errorformat+=%f:%l.%c-%*[0-9].%*[0-9]:\ %m

func! s:RunOracle(mode) abort
  " TODO(adonovan): support selections, not just positions.
  let s:pos = line2byte(line("."))+col(".")
  let s:errfile = tempname()
  let s:cmd = printf("!%s -mode=%s -pos=%s:#%d %s >%s",
    \ s:go_oracle, a:mode, bufname(""), s:pos, s:scope, s:errfile)
  execute s:cmd
  execute "cfile " . s:errfile
endfun

" Describe the expression at the current point.
command! GoOracleDescribe
  \ call s:RunOracle("describe")

" Show possible callees of the function call at the current point.
command! GoOracleCallees
  \ call s:RunOracle("callees")

" Show the set of callers of the function containing the current point.
command! GoOracleCallers
  \ call s:RunOracle("callers")

" Show the callgraph of the current program.
command! GoOracleCallgraph
  \ call s:RunOracle("callgraph")

" Describe the 'implements' relation for types in the
" package containing the current point.
command! GoOracleImplements
  \ call s:RunOracle("implements")

" Enumerate the set of possible corresponding sends/receives for
" this channel receive/send operation.
command! GoOracleChannelPeers
  \ call s:RunOracle("peers")
