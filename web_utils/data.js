    var chart;
    $.ajaxSetup({
        cache: false
    });
    var labelData = [];
    var trainingValueData = [];
    var current_training_loss;

    function initGraph() {
        $.get('web_utils/train.txt', function(data) {
            labelData.length = 0;
            trainingValueData.length = 0;
            var previous;
            var lines = data.split("\n");
            for (var i = 0, len = lines.length; i < len; i++) {
                if (i == 0) {
                    continue;
                }
                if (i == 1) {
                    var value = lines[i].split(":")[1];
                    previous = value;
                    labelData.push("");
                    trainingValueData.push(value);
                } else {
                    var epoch = lines[i].split(":")[0];
                    var value = lines[i].split(":")[1];
                    if (i + 1 == lines.length) {
                        current_training_loss = value;
                        $(".training_loss").text(current_training_loss);
                    }
                    if (((previous + 0.01) < value) || ((previous - 0.01) > value)) { // Check if previous data point is within 0.01 of next one
                        previous = value;
                        continue;

                    } else {
                        previous = value;
                        labelData.push(epoch);
                        trainingValueData.push(value);
                    }
                }
            }
        }, 'text');
    }

    function graph() {
        initGraph();
        var data = [];
        for (var i = 0; i < labelData.length; i++) {
            var object = {};
            object["x"] = Number(labelData[i]);
            object["y"] = Number(trainingValueData[i]);
            data.push(object);

        }
        chart = new CanvasJS.Chart("chartContainer", {

            data: [{
                type: "line",

                dataPoints: data

            }]
        });

        chart.render();
    }
    $('#circle').circleProgress({
        value: 0.00,
        size: 280,
        fill: {
            gradient: ["#00688B", "#00B2EE"]
        }
    });

    function update() {
        $.get('web_utils/data.txt', function(data) {
            var lines = data.split("\n");
            var iter;
            for (var i = 0, len = lines.length; i < len; i++) {
                if (i == 1) {
                    $('.current_epoch').text(lines[i]); // Update current epoch
                    var decimal = lines[i].split('.')[1]
                    if (decimal.length == 2) { // Sometimes the training script outputs epoch values with less than 3 decimal places
                        decimal = decimal + 0; 
                    } else if (decimal.length == 1) {
                        decimal = decimal + 00;
                    }
                    var percent = decimal / 1000;
                    $('#circle').circleProgress('value', decimal / 1000);

                } else if (i == 2) {
                    $('.current_iter').text(lines[i]); // Update current iteration
                    iter = lines[i];
                } else if (i == 3) {
                    var percent = Math.round((iter / lines[i]) * 10000) / 100;
                    $('#bar').text(percent + '%')
                    $('#bar').attr('aria-valuenow', percent);
                    $('#bar').css("width", percent + "%")
                } else if (i == 4) {
                    $('.time_per_batch').text(lines[i] + "s"); // Update time per batch
                }
            }
        }, 'text');
    }
    $(window).load(function() {
        update();
    });
    window.setInterval(function() {
        update();
    }, 3000);
    window.setInterval(function() {
        graph();
    }, 4000);