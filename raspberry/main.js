var express = require('express')
var app = express()

var wpi = require('wiring-pi');
 
//wpi.setup('wpi');
wpi.wiringPiSetupSys()


app.get('/', function (req, res) {

    var doorPin = 18;
    var greenPin = 18;
    var redPin = 4;

    var value = wpi.digitalRead(doorPin)

    console.log('Curr value ' + value)
    console.log('Request ' + req.query.open)
    open = req.query.open


    if (open == '1') {
    	console.log('Opening')
    	//wpi.digitalWrite(doorPin, 0)
    	wpi.digitalWrite(redPin, 0)
    	wpi.digitalWrite(greenPin, 1)
    }
    else {
    	console.log('Closing')
    	//wpi.digitalWrite(doorPin, 1)
    	wpi.digitalWrite(redPin, 1)
    	wpi.digitalWrite(greenPin, 0)
    }
        
    value = wpi.digitalRead(doorPin)
    console.log('New value ' + value)
    res.send('Hello World: ' + value)

})

app.listen(3000, function () {



  console.log('Example app listening on port 3000!')
})


function sleep(seconds){
    var waitUntil = new Date().getTime() + seconds*1000;
    while(new Date().getTime() < waitUntil) true;
}

console.log("Started...");
