" -*- text -*-
"  guru.vim -- Vim integration for the Go guru.
"
"  Load with (e.g.)  :source guru.vim
"  Call with (e.g.)  :GoGuruDescribe
"  while cursor or selection is over syntax of interest.
"  Run :copen to show the quick-fix file.
"
" This is an absolutely rudimentary integration of the Go Guru into
" Vim's quickfix mechanism and it needs a number of usability
" improvements before it can be practically useful to Vim users.
" Voluntary contributions welcomed!
"
" TODO(adonovan):
" - reject buffers with no filename.
" - hide all filenames in quickfix buffer.

" Get the path to the Go guru executable.
func! s:go_guru_bin()
  let [ext, sep] = (has('win32') || has('win64') ? ['.exe', ';'] : ['', ':'])
  let go_guru = globpath(join(split($GOPATH, sep), ','), '/bin/guru' . ext)
  if go_guru == ''
    let go_guru = globpath($GOROOT, '/bin/guru' . ext)
  endif
  return go_guru
endfunction

let s:go_guru = s:go_guru_bin()

func! s:qflist(output)
  let qflist = []
  " Parse GNU-style 'file:line.col-line.col: message' format.
  let mx = '^\(\a:[\\/][^:]\+\|[^:]\+\):\(\d\+\):\(\d\+\):\(.*\)$'
  for line in split(a:output, "\n")
    let ml = matchlist(line, mx)
    " Ignore non-match lines or warnings
    if ml == [] || ml[4] =~ '^ warning:'
      continue
    endif
    let item = {
    \  'filename': ml[1],
    \  'text': ml[4],
    \  'lnum': ml[2],
    \  'col': ml[3],
    \}
    let bnr = bufnr(fnameescape(ml[1]))
    if bnr != -1
      let item['bufnr'] = bnr
    endif
    call add(qflist, item)
  endfor
  call setqflist(qflist)
  cwindow
endfun

func! s:getpos(l, c)
  if &encoding != 'utf-8'
    let buf = a:l == 1 ? '' : (join(getline(1, a:l-1), "\n") . "\n")
    let buf .= a:c == 1 ? '' : getline('.')[:a:c-2]
    return len(iconv(buf, &encoding, 'utf-8'))
  endif
  return line2byte(a:l) + (a:c-2)
endfun

func! s:RunGuru(mode, selected) range abort
  let fname = expand('%:p')
  let sname = get(g:, 'go_guru_scope_file', fname)
  if a:selected != -1
    let pos1 = s:getpos(line("'<"), col("'<"))
    let pos2 = s:getpos(line("'>"), col("'>"))
    let cmd = printf('%s -scope=\'%s\' %s %s:#%d,#%d',
      \  s:go_guru,
      \  shellescape(fname), shellescape(sname), a:mode, pos1, pos2)
  else
    let pos = s:getpos(line('.'), col('.'))
    let cmd = printf('%s -scope=\'%s\' %s %s:#%d',
      \  s:go_guru,
      \  shellescape(fname), shellescape(sname), a:mode, pos)
  endif
  call s:qflist(system(cmd))
endfun

" Describe the expression at the current point.
command! -range=% GoGuruDescribe
  \ call s:RunGuru('describe', <count>)

" Show possible callees of the function call at the current point.
command! -range=% GoGuruCallees
  \ call s:RunGuru('callees', <count>)

" Show the set of callers of the function containing the current point.
command! -range=% GoGuruCallers
  \ call s:RunGuru('callers', <count>)

" Show the callgraph of the current program.
command! -range=% GoGuruCallgraph
  \ call s:RunGuru('callgraph', <count>)

" Describe the 'implements' relation for types in the
" package containing the current point.
command! -range=% GoGuruImplements
  \ call s:RunGuru('implements', <count>)

" Enumerate the set of possible corresponding sends/receives for
" this channel receive/send operation.
command! -range=% GoGuruChannelPeers
  \ call s:RunGuru('peers', <count>)
