##Instructions for enabling Go syntax highlighting in Fraise.app##
1. Move go.plist to /Applications/Fraise.app/Contents/Resources/Syntax\ Definitions/
2. Open /Applications/Fraise.app/Contents/Resources/SyntaxDefinitions.plist and add

	<dict>
		<key>name</key>
		<string>GoogleGo</string>
		<key>file</key>
		<string>go</string>
		<key>extensions</key>
		<string>go</string>
	</dict>
	
before </array>

3. Restart Fraise and you're good to Go!