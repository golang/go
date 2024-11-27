## Changes to the language {#language}

<!-- go.dev/issue/46477 -->
Go 1.24 now fully supports [generic type aliases](/issue/46477): a type alias
may be parameterized like a defined type.
See the [language spec](/ref/spec#Alias_declarations) for details.
For now, the feature can be disabled by setting `GOEXPERIMENT=noaliastypeparams`;
but the `aliastypeparams` setting will be removed for Go 1.25.
