var data = [];
var labels = [];

function init() {
    fetch('./data.json')
        .then(function (response) {
            return response.json();
        })
        .then(function (data) {
            df = data;
            var chartdata = [];
            var labels = [];

            var lossdata = [];
            var accdata = [];

            var vallossdata = [];
            var valaccdata = [];

            var batchlabels = [];

            var rand = Math.floor(Math.random() * 99) + 1;

            for (var i = 0; i < 187; i++) {
                chartdata.push(df["data"][rand][i])
                labels.push(i)
            }

            for (var i = 0; i < 15; i++) {
                lossdata.push(df["loss"][i])
                accdata.push(df["acc"][i])
                vallossdata.push(df["val_loss"][i])
                valaccdata.push(df["val_acc"][i])
                batchlabels.push(i)
            }

            var prediction = df["data"][rand][187]
            var actual = df["data"][rand][188]

            var mapping = {
                0: 'Sinus Rhythm',
                1: 'Ventricular',
                2: 'Supraventricular',
                3: 'Fusion Beat',
                4: 'Unclassified'
            }

            if (prediction == actual) {
                setColour(true);
            } else {
                setColour(false);
            }

            document.getElementById('prediction').innerText = 'Prediction: ' + mapping[prediction]
            document.getElementById('actual').innerText = 'Actual: ' + mapping[actual]
            document.getElementById('accuracy').innerText = 'Model Accuracy: ' + df["val_acc"][14].toFixed(3);

            training.data.datasets[0].data = lossdata;
            training.data.datasets[1].data = accdata;
            training.data.labels = batchlabels;

            validation.data.datasets[0].data = vallossdata;
            validation.data.datasets[1].data = valaccdata;
            validation.data.labels = batchlabels;

            chart.data.datasets[0].data = chartdata;
            chart.data.labels = labels;

            chart.update();
            training.update();
            validation.update();
        });
    
    chart.options.onClick = function (e) {
        var activePoints = chart.getElementsAtEvent(e);
        if (activePoints.length > 0) {
            var clickedElementIndex = activePoints[0]["_index"];
            var label = chart.data.labels[clickedElementIndex];
        } else {
            init();
        }
    }

    training.options.title.display = true;
    training.options.title.text = 'Training (Accuracy & Loss)'
    validation.options.title.display = true;
    validation.options.title.text = 'Validation (Accuracy & Loss)'

}

function setColour(boolean) {
    if (boolean) {
        chart.data.datasets[0].borderColor = '#388a24';
        chart.data.datasets[0].backgroundColor = '#388a24';
    } else {
        chart.data.datasets[0].borderColor = '#990b2e';
        chart.data.datasets[0].backgroundColor = '#990b2e';
    }
}

var ctx1 = document.getElementById('chart').getContext('2d');
var ctx2 = document.getElementById('training').getContext('2d');
var ctx3 = document.getElementById('validation').getContext('2d');

var opt = {
    title: {
        display: true,
    },
    legend: false,
    responsive: true,
    tooltips: {
        enabled: true,
    },
    hover: {
        mode: 'nearest',
        intersect: true
    },
    scales: {
        xAxes: [{
            display: false,
            scaleLabel: {
                display: true,
            }
        }],
        yAxes: [{
            display: true,
            scaleLabel: {
                display: true,
            }
        }]
    },
    elements: {
        point: {
            radius: 0
        }
    }
}

var chart = new Chart(ctx1, {
    type: 'line',
    data: {
        labels: labels,
        datasets: [{
            data: data,
            fill: false
        }],
    },
    options: opt,
});

var training = new Chart(ctx2, {
    type: 'line',
    data: {
        labels: labels,
        datasets: [{
            data: data,
            fill: false,
            borderColor: '#2c6bb8',
            backgroundColor: '#2c6bb8',
        }, {
            data: data,
            fill: false,
            borderColor: '#fac832',
            backgroundColor: '#fac832',
        }],
    },
    options : opt,
});
var validation = new Chart(ctx3, {
    type: 'line',
    data: {
        labels: labels,
        datasets: [{
            data: data,
            fill: false,
            borderColor: '#2c6bb8',
            backgroundColor: '#2c6bb8',
        }, {
            data: data,
            fill: false,
            borderColor: '#fac832',
            backgroundColor: '#fac832',
        }],
    },
    options: opt,
});

init();