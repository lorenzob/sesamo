# sesamo

folder web-identities contains all the acquired pictures for all identities.
svn-definition.pki is the generated fingerprint for all identities

enrollment-server: to acquire images for a new identity. GUI index.html
training-server: to transform pictures in the fingerprint. No GUI
recognition-server: to recognize known and unknown people on camera and send signals to a remote API that does something with the recognized identity. GUI: recog-index.html