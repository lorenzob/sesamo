/*
Copyright 2015-2016 Carnegie Mellon University

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

navigator.getUserMedia = navigator.getUserMedia ||
    navigator.webkitGetUserMedia ||
    navigator.mozGetUserMedia ||
    navigator.msGetUserMedia;

window.URL = window.URL ||
    window.webkitURL ||
    window.msURL ||
    window.mozURL;

// http://stackoverflow.com/questions/6524288
$.fn.pressEnter = function(fn) {

    return this.each(function() {
        $(this).bind('enterPress', fn);
        $(this).keyup(function(e){
            if(e.keyCode == 13)
            {
              $(this).trigger("enterPress");
            }
        })
    });
 };

function registerHbarsHelpers() {
    // http://stackoverflow.com/questions/8853396
    Handlebars.registerHelper('ifEq', function(v1, v2, options) {
        if(v1 === v2) {
            return options.fn(this);
        }
        return options.inverse(this);
    });
}

function sendFrameLoop() {
    if (socket == null || socket.readyState != socket.OPEN ||
        !vidReady || numNulls != defaultNumNulls) {
        return;
    }

    if (tok > 0) {
        var canvas = document.createElement('canvas');
        canvas.width = vid.width;
        canvas.height = vid.height;
        var cc = canvas.getContext('2d');
        cc.drawImage(vid, 0, 0, vid.width, vid.height);
        var apx = cc.getImageData(0, 0, vid.width, vid.height);

        var dataURL = canvas.toDataURL('image/jpeg', 0.6)

        var identity = "";
        if (training) {
            identity = $("#addPersonTxt").val();
        }
        
        var msg = {
            'type': 'FRAME',
            'dataURL': dataURL,
            'identity': identity
        };
        
    	socket.send(JSON.stringify(msg));
    	tok--;
    }
    setTimeout(function() {requestAnimFrame(sendFrameLoop)}, 250);
}


function getPeopleInfoHtml() {
    var info = {'-1': 0};
    var len = people.length;
    for (var i = 0; i < len; i++) {
        info[i] = 0;
    }

    var len = images.length;
    for (var i = 0; i < len; i++) {
        id = images[i].identity;
        info[id] += 1;
    }

    var h = "<li><b>Unknown:</b> "+info['-1']+"</li>";
    var len = people.length;
    for (var i = 0; i < len; i++) {
        h += "<li><b>"+people[i]+":</b> "+info[i]+"</li>";
    }
    return h;
}

function redrawPeople() {
    var context = {people: people, images: images};
    //$("#peopleTable").html(peopleTableTmpl(context));

    var context = {people: people};
    //$("#defaultPersonDropdown").html(defaultPersonTmpl(context));

    //$("#peopleInfo").html(getPeopleInfoHtml());
}

function getDataURLFromRGB(rgb) {
    var rgbLen = rgb.length;

    var canvas = $('<canvas/>').width(96).height(96)[0];
    var ctx = canvas.getContext("2d");
    var imageData = ctx.createImageData(96, 96);
    var data = imageData.data;
    var dLen = data.length;
    var i = 0, t = 0;

    for (; i < dLen; i +=4) {
        data[i] = rgb[t+2];
        data[i+1] = rgb[t+1];
        data[i+2] = rgb[t];
        data[i+3] = 255;
        t += 3;
    }
    ctx.putImageData(imageData, 0, 0);

    return canvas.toDataURL("image/png");
}

// non utilizzato nell'html
function updateRTT() {
    var diffs = [];
    for (var i = 5; i < defaultNumNulls; i++) {
        diffs.push(receivedTimes[i] - sentTimes[i]);
    }
    $("#rtt-"+socketName).html(
        jStat.mean(diffs).toFixed(2) + " ms (σ = " +
            jStat.stdev(diffs).toFixed(2) + ")"
    );
}

function sendState() {
    var msg = {
        'type': 'ALL_STATE',
        'images': images,
        'people': people,
        'training': training
    };
    socket.send(JSON.stringify(msg));
}

var canvas = null;

function createSocket(address, name) {
    socket = new WebSocket(address);
    socketName = name;
    socket.binaryType = "arraybuffer";
    socket.onopen = function() {
        $("#serverStatus").html("Connected to " + name);
        sentTimes = [];
        receivedTimes = [];
        // tok sblocca la spedizione di un altro frame dopo che quello
        // precedente e' stato processato. Se > 0 spedisce
        tok = defaultTok;  // = 1
        numNulls = 0

        socket.send(JSON.stringify({'type': 'NULL'}));
        sentTimes.push(new Date());
    }
    socket.onmessage = function(e) {
        console.log(e);
        j = JSON.parse(e.data)
        // una specie di "handshake per vedere che la connessione
        // sia stabile???
        if (j.type == "NULL") {
            receivedTimes.push(new Date());
            numNulls++;
            if (numNulls == defaultNumNulls) {
                updateRTT();
                sendState();
                sendFrameLoop();
            } else {
                socket.send(JSON.stringify({'type': 'NULL'}));
                sentTimes.push(new Date());
            }
        } else if (j.type == "PROCESSED") {
        	// il server mi dice che ha finito e riabilito la spedizione
        	// del prossimo frame
            tok++;
        } else if (j.type == "NEW_IMAGE") {
        	/*
        	 * Il server mi spedisce una delle immaginine di training
        	 * da mettere in basso (cropped and aligned)
        	 */
            images.push({
                hash: j.hash,
                identity: j.identity,
                image: getDataURLFromRGB(j.content),
                representation: j.representation
            });
            redrawPeople();
        } else if (j.type == "IDENTITIES") {
        	/*
        	 * Elenco delle persone riconosciute
        	 */
            var h = "Last updated: " + (new Date()).toTimeString();
            h += "<ul>";
            var len = j.identities.length
            if (len > 0) {
                for (var i = 0; i < len; i++) {
                    var identity = j.identities[i];
                    /*
                    var idIdx = j.identities[i];
                    if (idIdx != -1) {
                        identity = people[idIdx];
                    }
                    */
                    h += "<li>" + identity + "</li>";
                }
            } else {
                h += "<li>Nobody detected.</li>";
            }
            h += "</ul>"
            $("#peopleInVideo").html(h);
        } else if (j.type == "ANNOTATED") {
        	/*
        	 * Immagine con disegnati sopra i riferimenti del riconoscimento 
        	 * scritta dentro il div della webcam
        	 */
            $("#detectedFaces").html(
                "<img src='" + j['content'] + "' width='430px'></img>"
            )
        } else if (j.type == "TSNE_DATA") {
        	/*
        	 * Dovrebbe visualizzare le cose in modo diverso ma non sembra
        	 * funzionare...
        	 */
            BootstrapDialog.show({
                message: "<img src='" + j['content'] + "' width='100%'></img>"
            });
        } else if (j.type == "MATCHES") {
            
        	ids = j.identities
        	drawBoxes(ids);
        	
        } else {
            console.log("Unrecognized message type: " + j.type);
        }
    }
    socket.onerror = function(e) {
        console.log("Error creating WebSocket connection to " + address);
        console.log(e);
    }
    socket.onclose = function(e) {
        if (e.target == socket) {
            $("#serverStatus").html("Disconnected.");
        }
    }
}

function resetBoundingBoxCanvas() {
	if (canvas != null) {
		tmpCanvas = canvas
		canvas = null
		var context = tmpCanvas.getContext('2d');
		context.clearRect(0, 0, tmpCanvas.width, tmpCanvas.height);            
	}
}

function stringToColour(str) {
    for (var i = 0, hash = 0; i < str.length; hash = str.charCodeAt(i++) + ((hash << 5) - hash));
    color = Math.floor(Math.abs((Math.sin(hash) * 10000) % 1 * 16777216)).toString(16);
    return '#' + Array(6 - color.length + 1).join('0') + color;
}

function umSuccess(stream) {
    if (vid.mozCaptureStream) {
        vid.mozSrcObject = stream;
    } else {
        vid.src = (window.URL && window.URL.createObjectURL(stream)) ||
            stream;
    }
    vid.play();
    vidReady = true;
    sendFrameLoop();
}

function drawBoxes(ids) {
	
    var len = ids.length

    if (canvas == null) {
    	
        canvas = document.createElement('canvas'); //Create a canvas element
        //Set canvas width/height
        canvas.style.width='100%';
        canvas.style.height='100%';
        //Set canvas drawing area width/height
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        //Position canvas
        canvas.style.position='absolute';
        
        canvas.style.left=0;
        canvas.style.top=0;
        canvas.style.zIndex=100000;
        canvas.style.pointerEvents='none'; //Make sure you can click 'through' the canvas
        document.body.appendChild(canvas); //Append canvas to body element
    }
    
    var pos = document.getElementById('videoel').getBoundingClientRect();
    var context = canvas.getContext('2d');
    context.clearRect(0, 0, canvas.width, canvas.height);            

    for (var i = 0; i < len; i++) {
        var subject = ids[i];
    
        var nome = subject[0]
        var conf = parseFloat(subject[1]).toFixed(3);
        var bbox = subject[2]
    	
        var x = bbox[0];
        var y = bbox[3];
        var width = bbox[2] - bbox[0];
        var height = bbox[1] - bbox[3];

        context.beginPath();
        context.lineWidth = 3;
        context.strokeStyle = stringToColour(nome);
        //context.strokeStyle = "LimeGreen" 
        context.strokeRect(x + pos.x, y + pos.y, width, height);
        
        context.font = "14px Arial";
        context.fillStyle = stringToColour(nome) 
        context.fillText(nome + " (" + conf + ")", x + pos.x, y + pos.y - 5);
        context.closePath();
    }
}

function addPersonCallback(el) {
    defaultPerson = people.length;
    var newPerson = $("#addPersonTxt").val();
    if (newPerson == "") return;
    people.push(newPerson);
    $("#addPersonTxt").val("");

    if (socket != null) {
        var msg = {
            'type': 'ADD_PERSON',
            'val': newPerson,
        };
        socket.send(JSON.stringify(msg));
    }
    redrawPeople();
}

function trainingChkCallback() {
    /*
     
    if (training) {
        makeTabActive("tab-preview");
    } else {
        makeTabActive("tab-annotated");
    }
    */
	training = $("#trainingChk").prop('checked');
	
    var newPerson = $("#addPersonTxt").val();
    if (training && newPerson.trim() == '') {
    	alert("Please specify person name for training.")
    	$("#trainingChk").bootstrapToggle('off');
    	return false;
    }
    
    if (socket != null) {
        var msg = {
            'type': 'TRAINING',
            'val': training,
            'extra': newPerson
        };
        socket.send(JSON.stringify(msg));
    }
}

function startTrainingCallback(el) {
    if (socket != null) {
        var msg = {
            'type': 'START_TRAINING',
        };
        socket.send(JSON.stringify(msg));
    }
}

function reloadNetworkDataCallback(el) {
    if (socket != null) {
        var msg = {
            'type': 'RELOAD_SVM',
        };
        socket.send(JSON.stringify(msg));
    }
}


function setTabLinks() {
	
	assignHref('goto-enrollment-link', 'index.html', 8001)
	assignHref('goto-training-link', 'training.html', 8002)
	assignHref('goto-recognition-link', 'recognition.html', 8003)
}

function assignHref(anchorId, page, port) {
	elem = document.getElementById(anchorId);
	// truccaccio per farne solo due
	if (elem != null) {
		elem.href = "http://" + window.location.hostname + ":" + port + "/" + page;
	}
}

function findImageByHash(hash) {
    var imgIdx = 0;
    var len = images.length;
    for (imgIdx = 0; imgIdx < len; imgIdx++) {
        if (images[imgIdx].hash == hash) {
            console.log("  + Image found.");
            return imgIdx;
        }
    }
    return -1;
}

function updateIdentity(hash, idx) {
    var imgIdx = findImageByHash(hash);
    if (imgIdx >= 0) {
        images[imgIdx].identity = idx;
        var msg = {
            'type': 'UPDATE_IDENTITY',
            'hash': hash,
            'idx': idx
        };
        socket.send(JSON.stringify(msg));
    }
}

function removeImage(hash) {
    console.log("Removing " + hash);
    var imgIdx = findImageByHash(hash);
    if (imgIdx >= 0) {
        images.splice(imgIdx, 1);
        redrawPeople();
        var msg = {
            'type': 'REMOVE_IMAGE',
            'hash': hash
        };
        socket.send(JSON.stringify(msg));
    }
}

function changeServerCallback() {
    $(this).addClass("active").siblings().removeClass("active");
    switch ($(this).html()) {
    case "Local":
        socket.close();
        redrawPeople();
        createSocket("ws:" + window.location.hostname + ":9000", "Local");
        break;
    case "CMU":
        socket.close();
        redrawPeople();
        createSocket("ws://facerec.cmusatyalab.org:9000", "CMU");
        break;
    case "AWS East":
        socket.close();
        redrawPeople();
        createSocket("ws://54.159.128.49:9000", "AWS-East");
        break;
    case "AWS West":
        socket.close();
        redrawPeople();
        createSocket("ws://54.188.234.61:9000", "AWS-West");
        break;
    default:
        alert("Unrecognized server: " + $(this.html()));
    }
}
