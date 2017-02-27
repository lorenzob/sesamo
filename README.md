# sesamo

folder web-identities contains all the acquired pictures for all identities.
svn-definition.pkl is the generated fingerprint for all identities

enrollment-server: to acquire images for a new identity. GUI index.html
training-server: to transform pictures in the fingerprint. GUI training.html
recognition-server: to recognize known and unknown people on camera and send signals to a remote API that does something with the recognized identity. GUI: recognition.html

Avvio: ./start-all.sh

Check con: netstat -an |grep LIST | head

Log (di tutti): tail -f nohup.out

Shutdown:  killall python2


--> http://HOST:8001/

Singoli server:

nohup ./enrollment-server.sh &
nohup ./training-server.sh &
nohup ./recognition-server.sh &
