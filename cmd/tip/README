============================================================
Old instructions, only valid for talks.golang.org:
============================================================

1. Deploy the app.

   To deploy tip.golang.org:
       (See Kubernetes instruction below.)

   To deploy talks.golang.org:
       $ gcloud --project golang-org app deploy --no-promote talks.yaml

2. Wait until the deployed version is serving requests.

3. Go to the developer console and upgrade the default version.
   https://console.developers.google.com/appengine/versions?project=golang-org&moduleId=tip

4. Clean up any old versions (they continue to use at least one instance).

============================================================
New Kubernetes instructions, for tip.golang.org:
============================================================

Kubernetes instructions:

 * build & push images (see Makefile for helpers)
 * create/update resources:
   - kubectl create -f tip-rc.yaml
   - kubectl create -f tip-service.yaml

TODO(bradfitz): flesh out these instructions as I gain experience
with updating this over time. Also: move talks.golang.org to GKE too?
