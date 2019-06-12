function info(message) {
    console.info(`[%cinfo%c] ${message}`, "color: cyan; font-weight: bold", "color: inherit; font-weight: initial");
}

function ok(message) {
    console.info(`[%cok%c] ${message}`, "color: #0f0; font-weight: bold", "color: inherit; font-weight: initial");
}

const minVal = [15.2, 15.62, 14.87, 15.15, 3458100.0];
const maxVal = [178.94, 180.38, 175.75, 179.94, 591052200.0];

info("Loading model...");
const modelPromise = tf.loadLayersModel("src/rnn_test/rnn_model_js/model.json").then(model => {
    window.model = model;
    ok("Loaded model.");
    return model;
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

async function predict(data, days, cb) {
    info("Trimming and scaling data...");
    let trimmedData = data.slice(-240).map(row => row.slice(1).map((val, idx) => {
        return (val - minVal[idx]) / (maxVal[idx] - minVal[idx]);
    }));

    info("Running model...");
    let i = 0;
    function run(done) {
        console.log(`Running model (${i + 1}/${days})...`);
        const input = tf.tensor([trimmedData.slice(-240).flat()]);
        const output = model.predict(input).dataSync();

        trimmedData.push(Array.from(output));
        cb(data.length + i, unscale(output));
        i++;

        if (i < days) {
            setTimeout(() => run(done), 1);
        } else {
            done();
        }
    }

    await new Promise(res => {
       run(res);
    });

    ok("Done predicting.");
    return trimmedData.slice(240).map(unscale);
}

Vue.component("sdf-stock", {
    props: ["symbol"],
    data() {
        return {
            isLoaded: false,
            error: false,
            chart: null,
            data: null,
            featureIndex: 1
        };
    },
    mounted() {
        info(`Loading data for ${this.$props.symbol}...`);

        fetch(`src/rnn_test/data/${this.$props.symbol}.csv`).then(async response => {
            if (!response.ok) {
                throw new Error(response.statusText);
            }

            this.isLoaded = true;

            ok(`Loaded data for ${this.$props.symbol}.`);
            info(`Parsing data for ${this.$props.symbol}...`);

            const text = await response.text();
            this.data = text.split("\n").slice(1).map(line => {
                const cols = line.split(",");

                const date = new Date(cols[0] + " GMT");

                return [date.getTime() / (24 * 60 * 60 * 1000), ...cols.slice(1).map(parseFloat)];
            }).filter(row => row.length === 6);

            info(`Creating chart for ${this.$props.symbol}...`);

            const graphLookBack = 3128;
            const chartData = {
                datasets: [{
                    data: this.data.slice(-graphLookBack).map((line, index) => {return {x: index + this.data.length - graphLookBack, y: line[this.featureIndex + 1]}}),
                    xAxisId: "x-axis",
                    label: "Actual",
                    borderColor: "rgb(24, 128, 255)",
                    backgroundColor: "rgba(24, 128, 255, 0.2)",
                    borderWidth: 2
                }, {
                    data: [],
                    xAxisId: "x-axis",
                    label: "Predicted",
                    borderColor: "rgb(24, 170, 128)",
                    backgroundColor: "rgba(24, 170, 128, 0.2)",
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
                            position: "bottom"
                        }]
                    },
                    elements: {
                        point: {radius: 0}
                    }
                }
            });

            ok(`Parsed data and created chart for ${this.$props.symbol}.`);

            const days = 365;
            info(`Predicting over ${days} days - ${this.$props.symbol}...`);
            await predict(this.data, days, (index, data) => {
                this.chart.data.datasets[1].data.push({x: index, y: data[this.featureIndex]});
                this.chart.update();
            });
        }).catch(e => {
            console.log(e);
            this.error = true;
        });
    },
    computed: {
        canvasStyle() {
            return this.isLoaded ? {display: "block"} : {display: "none"};
        }
    },
    // language=HTML
    template: `
        <div>
          <hr />
          <div class="row" >
            <div class="col-md-2"><h3>{{symbol}}</h3></div>
            <div class="col">
              <canvas ref="chartCanvas" class="w-100 chart" v-bind:style="canvasStyle"></canvas>
              <span class="text-danger" v-if="error">Failed to load stock data</span>
            </div>
          </div>
        </div>
    `,

});

const app = new Vue({
    el: "#app",
    data: function () {
        return {
            ticker: "AAPL",
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
                event.target.reset();
                this.tickers.push(this.ticker.toUpperCase());
            }
        }
    }
});