function info(message) {
    console.info(`[%cinfo%c] ${message}`, "color: cyan; font-weight: bold", "color: inherit; font-weight: initial");
}

function ok(message) {
    console.info(`[%cok%c] ${message}`, "color: #0f0; font-weight: bold", "color: inherit; font-weight: initial");
}

const minVal = [15.2, 15.62, 14.87, 15.15, 3458100.0];
const maxVal = [178.94, 180.38, 175.75, 179.94, 591052200.0];

let alphaVantageApiKey;

info("Loading model...");
const modelPromise = tf.loadLayersModel("src/rnn_test/rnn_model_js/model.json").then(model => {
    window.model = model;
    ok("Loaded model.");
    return model;sw3
});

async function getModel() {
    if (window.model) {
        return window.model;
    } else {
        return await modelPromise;
    }
}

function unscale(row) {
    return row.map((val, idx) => {
        return val * (maxVal[idx] - minVal[idx]) + minVal[idx];
    });
}

function predict(data, cb, timeout = 1) {
    info("Trimming and scaling data...");
    let trimmedData = data.slice(-240).map(row => row.map((val, idx) => {
        return (val - minVal[idx]) / (maxVal[idx] - minVal[idx]);
    }));

    info("Running model...");
    function run() {
        console.log(`Running model...`);
        const input = tf.tensor([trimmedData.slice(-240).flat()]);
        const output = model.predict(input).dataSync();

        trimmedData.push(Array.from(output));
        if (!cb(unscale(output))) {
            setTimeout(run, timeout);
        }
    }

    run();
}

Vue.component("sdf-stock", {
    props: ["symbol"],
    data() {
        return {
            error: false,
            chart: null,
            data: null,
            predicted: [],
            featureIndex: 1,
            isPredicting: false
        };
    },
    mounted() {
        this.reload();
    },
    methods: {
        async reload() {
            this.error = false;
            try {
                info(`Loading data for ${this.$props.symbol}...`);

                let response = await fetch(`src/rnn_test/data/${this.$props.symbol}.csv`);
                if (response.status === 404) {
                    response = await fetch(`https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=${this.$props.symbol}&apikey=${alphaVantageApiKey}&outputsize=full&datatype=csv`)
                }

                if (!response.ok) {
                    throw new Error(response.statusText);
                }

                const text = await response.text();

                ok(`Loaded data for ${this.$props.symbol}.`);
                info(`Parsing data for ${this.$props.symbol}...`);

                this.data = text.split("\n").slice(1).map(line => {
                    const cols = line.split(",");
                    return cols.slice(1).map(parseFloat);
                }).filter(row => row.length === 5);

                info(`Creating chart for ${this.$props.symbol}...`);

                const graphLookBack = 720;
                const chartData = {
                    datasets: [{
                        data: this.data.slice(-graphLookBack).map((line, index) => {
                            return {
                                x: index + this.data.length - graphLookBack,
                                y: line[this.featureIndex + 1]
                            }
                        }),
                        xAxisId: "x-axis",
                        label: "Actual",
                        borderColor: "rgb(24, 128, 255)",
                        backgroundColor: "rgba(24, 128, 255, 0.2)",
                        borderWidth: 2
                    }, {
                        data: [],
                        xAxisId: "x-axis",
                        label: "Predicted",
                        borderColor: "rgb(128, 128, 128)",
                        backgroundColor: "rgba(128, 128, 128, 0.2)",
                        borderWidth: 2
                    }]
                };

                this.chart = new Chart(this.$refs.chartCanvas.getContext("2d"), {
                    type: "line",
                    data: chartData,
                    options: {
                        scales: {
                            xAxes: [{
                                id: "x-axis",
                                type: "linear",
                                position: "bottom",
                                scaleLabel: {
                                    display: true,
                                    labelString: "Data Index"
                                }
                            }],
                            yAxes: [{
                                id: "y-axis",
                                type: "linear",
                                position: "left",
                                gridLines: {
                                    color: "rgba(255, 255, 255, 0.1)",
                                    zeroColor: "rgba(255, 255, 255, 0.2)"
                                },
                                scaleLabel: {
                                    display: true,
                                    labelString: "Share Price"
                                },
                                ticks: {
                                    callback: (value, index, values) => {
                                        return "$" + value;
                                    }
                                }
                            }]
                        },
                        tooltips: {
                            callbacks: {
                                title() {},
                                label: (item, data) => {
                                    let label = "$" + Math.round(item.yLabel * 100) / 100;
                                    if (label.split(".")[1].length < 2) {
                                        label += "0";
                                    }
                                    return label;
                                }
                            }
                        },
                        elements: {
                            point: {radius: 0}
                        }
                    }
                });

                ok(`Parsed data and created chart for ${this.$props.symbol}.`);
            } catch (e) {
                console.error(e);
                this.error = e;
            }
        },
        startPredicting(days, offset = 0) {
            if (!this.isPredicting) {
                this.isPredicting = true;
                info(`Predicting over ${days} days - ${this.$props.symbol}...`);
                let i = 0;
                predict(this.data.slice(0, -(offset + 1)).concat(this.predicted), data => {
                    this.predicted.push(Array.from(data));
                    this.chart.data.datasets[1].data.push({x: this.data.length - offset + this.predicted.length - 1, y: data[this.featureIndex]});

                    const comparisonResult = this.data[this.data.length - 1][this.featureIndex] - data[this.featureIndex];
                    if (comparisonResult === 0) {
                        this.chart.data.datasets[1].borderColor = "rgb(128, 128, 128)";
                        this.chart.data.datasets[1].backgroundColor = "rgba(128, 128, 128, 0.2)";
                    } else if (comparisonResult > 0) {
                        this.chart.data.datasets[1].borderColor = "rgb(186, 42, 10)";
                        this.chart.data.datasets[1].backgroundColor = "rgba(186, 42, 10, 0.2)";
                    } else {
                        this.chart.data.datasets[1].borderColor = "rgb(10, 186, 128)";
                        this.chart.data.datasets[1].backgroundColor = "rgba(10, 186, 128, 0.2)";
                    }

                    this.chart.update({duration: 0});

                    i++;
                    if (i >= days) {
                        this.isPredicting = false;
                    }
                    return !this.isPredicting;
                });
            }
        },
        clear() {
            this.$emit("remove");
        }
    },
    // language=HTML
    template: `
      <div>
        <hr />
        <div class="row" >
          <div class="col-md-2">
            <h3>{{symbol}}</h3>
            <div v-if="data">
              <p v-if="isPredicting"><a href="#" @click.prevent="isPredicting = false">Stop Predicting</a></p>
              <div v-else>
                <p><a href="#" @click.prevent="startPredicting(30)">Predict 1 Month</a></p>
                <p><a href="#" @click.prevent="startPredicting(90)">Predict 3 Months</a></p>
                <p><a href="#" @click.prevent="startPredicting(180)">Predict 6 Months</a></p>
                <p><a href="#" @click.prevent="startPredicting(365)">Predict 1 Year</a></p>
                <p><a href="#" @click.prevent="startPredicting(Infinity)">Predict Forever</a></p>
                <p><a href="#" @click.prevent="startPredicting(-4002)">Simulate 2008 Prediction</a></p>
              </div>
            </div>
            <p v-else class="text-secondary">Loading data...</p>
            <p><a href="#" @click.prevent="clear()">Clear</a></p>
          </div>
          <div class="col">
            <canvas ref="chartCanvas" class="w-100 chart"></canvas>
            <div v-if="error">
              <p class="text-danger">Failed to load stock data. Try setting the Alpha Vantage API key or using a different symbol.</p>
              <p class="text-danger">Full error message: {{error.message}}</p>
              <p><a href="#" @click.prevent="reload()">Try again?</a></p>
            </div>
          </div>
        </div>
      </div>
    `,

});

const app = new Vue({
    el: "#app",
    data: function () {
        return {
            ticker: "AMZN",
            tickers: []
        }
    },
    mounted() {
        ok("Mounted Vue.")
    },
    methods: {
        addTicker(event) {
            event.preventDefault();
            if (!this.tickers.includes(this.ticker.toUpperCase())) {
                this.$refs.addTickerForm.reset();
                this.tickers.splice(0, 0, this.ticker.toUpperCase());
            }
        },
        getAPIKey() {
            alphaVantageApiKey = prompt("Enter the Alpha Vantage API key.") || alphaVantageApiKey;
        }
    }
});