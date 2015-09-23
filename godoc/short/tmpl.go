// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

package short

const templateHTML = `
<!doctype HTML>
<html>
<head>
<title>golang.org URL shortener</title>
<style>
body {
	background: white;
}
input {
	border: 1px solid #ccc;
}
input[type=text] {
	width: 400px;
}
input, td, th {
	color: #333;
	font-family: Georgia, Times New Roman, serif;
}
input, td {
	font-size: 14pt;
}
th {
	font-size: 16pt;
	text-align: left;
	padding-top: 10px;
}
.autoselect {
	border: none;
}
.error {
	color: #900;
}
table {
	margin-left: auto;
	margin-right: auto;
}
</style>
</head>
<body>

<table>

{{with .Error}}
<tr>
	<th colspan="3">Error</th>
</tr>
<tr>
	<td class="error" colspan="3">{{.}}</td>
</tr>
{{end}}

<tr>
	<th>Key</th>
	<th>Target</th>
	<th></th>
</tr>

<form method="POST" action="{{.Prefix}}">
<tr>
	<td><input type="text" name="key"{{with .New}} value="{{.Key}}"{{end}}></td>
	<td><input type="text" name="target"{{with .New}} value="{{.Target}}"{{end}}></td>
	<td><input type="submit" name="do" value="Add">
</tr>
</form>

{{with .Links}}
<tr>
	<th>Short Link</th>
	<th>&nbsp;</th>
	<th>&nbsp;</th>
</tr>
{{range .}}
<tr>
	<td><input class="autoselect" type="text" orig="{{$.BaseURL}}/{{.Key}}" value="{{$.BaseURL}}/{{.Key}}"></td>
	<td><input class="autoselect" type="text" orig="{{.Target}}" value="{{.Target}}"></td>
	<td>
		<form method="POST" action="{{$.Prefix}}">
			<input type="hidden" name="key" value="{{.Key}}">
			<input type="submit" name="do" value="Delete" class="delete">
		</form>
	</td>
</tr>
{{end}}
{{end}}

</table>

</body>
<script type="text/javascript" src="https://ajax.googleapis.com/ajax/libs/jquery/1.8.2/jquery.min.js"></script>
<script type="text/javascript">window.jQuery || document.write(unescape("%3Cscript src='/doc/jquery.js' type='text/javascript'%3E%3C/script%3E"));</script>
<script>
$(document).ready(function() {
	$('.autoselect').each(function() {
		$(this).click(function() {
			$(this).select();
		});
		$(this).change(function() {
			$(this).val($(this).attr('orig'));
		});
	});
	$('.delete').click(function(e) {
		var link = $(this).closest('tr').find('input').first().val();
		var ok = confirm('Delete this link?\n' + link);
		if (!ok) {
			e.preventDefault();
			return false;
		}
	});
});
</script>
</html>
`
