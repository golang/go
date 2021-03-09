# Security Policy

## Supported Versions

We support the past two Go releases (for example, Go 1.12.x and Go 1.13.x).

See https://golang.org/wiki/Go-Release-Cycle and in particular the
[Release Maintenance](https://github.com/golang/go/wiki/Go-Release-Cycle#release-maintenance)
part of that page.

## Reporting a Vulnerability

See https://golang.org/security for how to report a vulnerability.

## atlas-create-home-zip
This will create a file called generated-test-resources.zip in the plugins target folder.  Copy this file to a known location (such as, src / test / resources).  Finally add:
<productDataPath>${basedir}/src/test/resources/generated-test-resources.zip</productDataPath>
to the <configuration/> of the maven-jira-plugin or maven-confluence-plugin in your pom.xml. The completed configuration should look something like this:
 curl -v -X POST https://api-m.sandbox.paypal.com/v2/checkout/orders \
 -H "Content-Type: Application / json" \
 -H "Authorization: Holder Access Token" \
 D '{
   "Intent": "CAPTURE",
   "purchase_units": [
     {
       "Balance": {
         "currency_code": "USD",
         "Value": "100.00"
       )
     )
   )
 ) '
