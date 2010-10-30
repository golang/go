" Copyright 2009 The Go Authors. All rights reserved.
" Use of this source code is governed by a BSD-style
" license that can be found in the LICENSE file.
"
" go.vim: Vim syntax file for Go.
"
" Options:
"   There are some options for customizing the highlighting; the recommended
"   settings are the default values, but you can write:
"     let OPTION_NAME = 0
"   in your ~/.vimrc file to disable particular options. You can also write:
"     let OPTION_NAME = 1
"   to enable particular options. At present, all options default to on.
"
"   - go_highlight_array_whitespace_error
"     Highlights white space after "[]".
"   - go_highlight_chan_whitespace_error
"     Highlights white space around the communications operator that don't follow
"     the standard style.
"   - go_highlight_extra_types
"     Highlights commonly used library types (os.Error, etc.).
"   - go_highlight_space_tab_error
"     Highlights instances of tabs following spaces.
"   - go_highlight_trailing_whitespace_error
"     Highlights trailing white space.

" Quit when a (custom) syntax file was already loaded
if exists("b:current_syntax")
  finish
endif

if !exists("go_highlight_array_whitespace_error")
  let go_highlight_array_whitespace_error = 1
endif
if !exists("go_highlight_chan_whitespace_error")
  let go_highlight_chan_whitespace_error = 1
endif
if !exists("go_highlight_extra_types")
  let go_highlight_extra_types = 1
endif
if !exists("go_highlight_space_tab_error")
  let go_highlight_space_tab_error = 1
endif
if !exists("go_highlight_trailing_whitespace_error")
  let go_highlight_trailing_whitespace_error = 1
endif

syn case match

syn keyword     goDirective         package import
syn keyword     goDeclaration       var const type
syn keyword     goDeclType          struct interface

hi def link     goDirective         Statement
hi def link     goDeclaration       Keyword
hi def link     goDeclType          Keyword

" Keywords within functions
syn keyword     goStatement         defer go goto return break continue fallthrough
syn keyword     goConditional       if else switch select
syn keyword     goLabel             case default
syn keyword     goRepeat            for range

hi def link     goStatement         Statement
hi def link     goConditional       Conditional
hi def link     goLabel             Label
hi def link     goRepeat            Repeat

" Predefined types
syn keyword     goType              chan map bool string
syn keyword     goSignedInts        int int8 int16 int32 int64
syn keyword     goUnsignedInts      byte uint uint8 uint16 uint32 uint64 uintptr
syn keyword     goFloats            float float32 float64
syn keyword     goComplexes         complex complex64 complex128

hi def link     goType              Type
hi def link     goSignedInts        Type
hi def link     goUnsignedInts      Type
hi def link     goFloats            Type
hi def link     goComplexes         Type

" Treat func specially: it's a declaration at the start of a line, but a type
" elsewhere. Order matters here.
syn match       goType              /\<func\>/
syn match       goDeclaration       /^func\>/

" Predefined functions and values
syn keyword     goBuiltins          append cap close closed cmplx copy imag len
syn keyword     goBuiltins          make new panic print println real recover
syn keyword     goConstants         iota true false nil

hi def link     goBuiltins          Keyword
hi def link     goConstants         Keyword

" Comments; their contents
syn keyword     goTodo              contained TODO FIXME XXX BUG
syn cluster     goCommentGroup      contains=goTodo
syn region      goComment           start="/\*" end="\*/" contains=@goCommentGroup,@Spell
syn region      goComment           start="//" end="$" contains=@goCommentGroup,@Spell

hi def link     goComment           Comment
hi def link     goTodo              Todo

" Go escapes
syn match       goEscapeOctal       display contained "\\[0-7]\{3}"
syn match       goEscapeC           display contained +\\[abfnrtv\\'"]+
syn match       goEscapeX           display contained "\\x\x\{2}"
syn match       goEscapeU           display contained "\\u\x\{4}"
syn match       goEscapeBigU        display contained "\\U\x\{8}"
syn match       goEscapeError       display contained +\\[^0-7xuUabfnrtv\\'"]+

hi def link     goEscapeOctal       goSpecialString
hi def link     goEscapeC           goSpecialString
hi def link     goEscapeX           goSpecialString
hi def link     goEscapeU           goSpecialString
hi def link     goEscapeBigU        goSpecialString
hi def link     goSpecialString     Special
hi def link     goEscapeError       Error

" Strings and their contents
syn cluster     goStringGroup       contains=goEscapeOctal,goEscapeC,goEscapeX,goEscapeU,goEscapeBigU,goEscapeError
syn region      goString            start=+"+ skip=+\\\\\|\\"+ end=+"+ contains=@goStringGroup
syn region      goRawString         start=+`+ end=+`+

hi def link     goString            String
hi def link     goRawString         String

" Characters; their contents
syn cluster     goCharacterGroup    contains=goEscapeOctal,goEscapeC,goEscapeX,goEscapeU,goEscapeBigU
syn region      goCharacter         start=+'+ skip=+\\\\\|\\'+ end=+'+ contains=@goCharacterGroup

hi def link     goCharacter         Character

" Regions
syn region      goBlock             start="{" end="}" transparent fold
syn region      goParen             start='(' end=')' transparent

" Integers
syn match       goDecimalInt        "\<\d\+\([Ee]\d\+\)\?\>"
syn match       goHexadecimalInt    "\<0x\x\+\>"
syn match       goOctalInt          "\<0\o\+\>"
syn match       goOctalError        "\<0\o*[89]\d*\>"

hi def link     goDecimalInt        Integer
hi def link     goHexadecimalInt    Integer
hi def link     goOctalInt          Integer
hi def link     Integer             Number

" Floating point
syn match       goFloat             "\<\d\+\.\d*\([Ee][-+]\d\+\)\?\>"
syn match       goFloat             "\<\.\d\+\([Ee][-+]\d\+\)\?\>"
syn match       goFloat             "\<\d\+[Ee][-+]\d\+\>"

hi def link     goFloat             Float

" Imaginary literals
syn match       goImaginary         "\<\d\+i\>"
syn match       goImaginary         "\<\d\+\.\d*\([Ee][-+]\d\+\)\?i\>"
syn match       goImaginary         "\<\.\d\+\([Ee][-+]\d\+\)\?i\>"
syn match       goImaginary         "\<\d\+[Ee][-+]\d\+i\>"

hi def link     goImaginary         Number

" Spaces after "[]"
if go_highlight_array_whitespace_error != 0
  syn match goSpaceError display "\(\[\]\)\@<=\s\+"
endif

" Spacing errors around the 'chan' keyword
if go_highlight_chan_whitespace_error != 0
  " receive-only annotation on chan type
  syn match goSpaceError display "\(<-\)\@<=\s\+\(chan\>\)\@="
  " send-only annotation on chan type
  syn match goSpaceError display "\(\<chan\)\@<=\s\+\(<-\)\@="
  " value-ignoring receives in a few contexts
  syn match goSpaceError display "\(\(^\|[={(,;]\)\s*<-\)\@<=\s\+"
endif

" Extra types commonly seen
if go_highlight_extra_types != 0
  syn match goExtraType /\<bytes\.\(Buffer\)\>/
  syn match goExtraType /\<io\.\(Reader\|Writer\|ReadWriter\|ReadWriteCloser\)\>/
  syn match goExtraType /\<\(os\.Error\)\>/
  syn match goExtraType /\<reflect\.\w*\(Type\|Value\)\>/
  syn match goExtraType /\<unsafe\.Pointer\>/
endif

" Space-tab error
if go_highlight_space_tab_error != 0
  syn match goSpaceError display " \+\t"me=e-1
endif

" Trailing white space error
if go_highlight_trailing_whitespace_error != 0
  syn match goSpaceError display excludenl "\s\+$"
endif

hi def link     goExtraType         Type
hi def link     goSpaceError        Error

let b:current_syntax = "go"
