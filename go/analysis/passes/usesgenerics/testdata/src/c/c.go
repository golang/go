// want package:`features{typeDecl,funcDecl,typeSet,typeInstance,funcInstance}`

// Features funcDecl, typeSet, and funcInstance come from imported packages "a"
// and "b". These features are not directly present in "c".

package c

import (
	"a"
	"b"
)

type T[P b.Constraint] a.T[P]
