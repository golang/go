@@ -1,16 +1,13 @@
 package format //@format("package")
 
 import (
-	"runtime"
 	"fmt"
 	"log"
+	"runtime"
 )
 
 func hello() {
 
-
-
-
 	var x int //@diag("x", "LSP", "x declared but not used")
 }