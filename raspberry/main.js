var express = require('express')
var app = express()

var wpi = require('wiring-pi');
 
//wpi.setup('wpi');
wpi.wiringPiSetupSys()


app.get('/', function (req, res) {

    var pin = 18;

    var value = wpi.digitalRead(pin)

    console.log('Curr value ' + value)
    console.log('Request ' + req.query.open)
    open = req.query.open


    if (open == '1') {
    	console.log('Opening')
    	wpi.digitalWrite(pin, 0)
    }
    else {
    	console.log('Closing')
    	wpi.digitalWrite(pin, 1)
    }
        
    value = wpi.digitalRead(pin)
    console.log('New value ' + value)
    res.send('Hello World: ' + value)

})

app.listen(3000, function () {



  console.log('Example app listening on port 3000!')
})


var noble = require('noble');

noble.on('scanStart', function () {
  console.log('Scanning...')
});


noble.on('stateChange', function(state) {

  console.log('State is ' + state)

  if (state === 'poweredOn')
    noble.startScanning();
  else
    noble.stopScanning();
});

	noble.on('discover', function(peripheral) {

   	noble.stopScanning();

console.log('peripheral with ID ' + peripheral.id + ' found');

  console.log('peripheral discovered (' + peripheral.id +
              ' with address <' + peripheral.address +  ', ' + peripheral.addressType + '>,' +
              ' connectable ' + peripheral.connectable + ',' +
              ' RSSI ' + peripheral.rssi + ':');
  console.log('\thello my local name is:');
  console.log('\t\t' + peripheral.advertisement.localName);
  console.log('\tcan I interest you in any of the following advertised services:');
  console.log('\t\t' + JSON.stringify(peripheral.advertisement.serviceUuids));

  var serviceData = peripheral.advertisement.serviceData;
  if (serviceData && serviceData.length) {
    console.log('\there is my service data:');
    for (var i in serviceData) {
      console.log('\t\t' + JSON.stringify(serviceData[i].uuid) + ': ' + JSON.stringify(serviceData[i].data.toString('hex')));
    }
}
  if (peripheral.advertisement.manufacturerData) {
    console.log('\there is my manufacturer data:');
    console.log('\t\t' + JSON.stringify(peripheral.advertisement.manufacturerData.toString('hex')));
  }
  if (peripheral.advertisement.txPowerLevel !== undefined) {
    console.log('\tmy TX power level is:');
    console.log('\t\t' + peripheral.advertisement.txPowerLevel);
  }

	if (peripheral.advertisement.localName.indexOf('Pizza') >= 0) {
		peripheral.connect(function(error) {
		
			console.log('connected ' + error);

			peripheral.discoverServices([], function(error, services) {
				console.log("services " + services.length);
				var service = services[2];
				console.log("service: " + service.uuid);

				service.discoverCharacteristics([], function(error, characteristics) {
					var ch = characteristics[0];
					console.log("char.len: " + characteristics.length);
					console.log("char: " + ch.properties);

					ch.read(function(error, data) {
                              			if (error) {
							console.log("ch read error: " + error);
                              			}
                              			if (data) {
							var string = data.toString('hex');
							console.log("ch data: " + string + '<');
                              			}
						else {
							console.log("no data");
						}
					});

					const arr = new Uint8Array(1);

					arr[0] = 1;
					// Copies the contents of `arr`
					const buf1 = new Buffer(arr);

                                        ch.write(buf1, false, function(error) {
                                                if (error) {
                                                        console.log("ch read error: " + error);
                                                }
                                        });

                                        ch.read(function(error, data) {
                                                if (error) {
                                                        console.log("post ch read error: " + error);
                                                }
                                                if (data) {
                                                        console.log("post ch data: " + data[0]  + '<');
                                                        var string = data.toString('hex');
                                                        console.log("post ch data: " + string + '<');
                                                }
                                                else {
                                                        console.log("post no data");
                                                }
                                        });



					ch.discoverDescriptors(function(error, descriptors) {

						console.log("descriptors: " + descriptors[0].name);
						descriptors[0].writeValue(new Buffer(1), function(error) {
                              				if (error) {
								console.log("write error: " + error);
                              				}
						}); //--
						descriptors[0].readValue(function(error, data) {
                              				if (error) {
								console.log("read error: " + error);
                              				}
                              				if (data) {
								console.log("data: " + data);
                              				}
							else {
								console.log("no data");
							}
						}); //--



					});

				});
			});

			console.log("waiting...");
			//sleep(5);
			console.log("waiting done.");
	/*	
			peripheral.disconnect(function(error) {
		  	console.log('disconnected ' + error);
			});
	*/
		
		});
	}
	else {
   		console.log('ignoro ' + peripheral.advertisement.localName);
	}


  console.log("fine discover");
});

function sleep(seconds){
    var waitUntil = new Date().getTime() + seconds*1000;
    while(new Date().getTime() < waitUntil) true;
}

console.log("Started...");
