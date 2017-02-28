#!/bin/bash

time curl -s -D - -H "Ocp-Apim-Subscription-Key: 4a1eaaa0bea94618b34c3fdd8804a4e9" https://westus.api.cognitive.microsoft.com/spid/v1.0/identify?identificationProfileIds=609c2dfd-51ff-426d-9658-914ff49b599e\&shortAudio=true --data-binary "@ciao-lorenzo.wav"

url=$(cat tmp | sed 's/Operation-Location: //;s/.$//')

# echo "HHH $url HHH"

time curl -H "Ocp-Apim-Subscription-Key: 4a1eaaa0bea94618b34c3fdd8804a4e9" $url


