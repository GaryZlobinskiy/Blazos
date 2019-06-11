function info(message) {
    console.info(`[%cinfo%c] ${message}`, "color: cyan; font-weight: bold", "color: inherit; font-weight: initial");
}

function ok(message) {
    console.info(`[%cok%c] ${message}`, "color: #0f0; font-weight: bold", "color: inherit; font-weight: initial");
}

info("Loading model...");
const modelPromise = tf.loadLayersModel("src/rnn_test/rnn_model_js/model.json").then(model => {
    window.model = model;
    ok("Loaded model.");
    worker.postMessage({type: "model", model});
    return model;
});

async function getModel() {
    if (window.model) {
        return window.model;
    } else {
        return await modelPromise;
    }
}

info("Spawning web worker...");
window.worker = new Worker("js/worker.js");
window.worker.addEventListener("message", e => {
    info(`Received message of type ${e.data.type}`)
});

Vue.component("sdf-stock", {
    props: ["symbol"],
    data() {
        return {
            isLoaded: false,
            error: false,
            chart: null,
            data: null
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
            });

            info(`Creating chart for ${this.$props.symbol}...`);

            const chartData = {
                datasets: [{
                    data: this.data.map((line, index) => {return {x: index, y: line[2]}}),
                    xAxisId: "x-axis",
                    label: "Actual",
                    borderColor: "rgb(24, 128, 255)",
                    backgroundColor: "rgba(24, 128, 255, 0.2)",
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

            info(`Waiting for model - ${this.$props.symbol}...`);
            await getModel();

            info(`Calling service worker - ${this.$props.symbol}...`);
            worker.postMessage({type: "predict", symbol: this.$props.symbol, data: this.data.slice(-60).map(row => row.slice(1)), days: 365});

            const handler = e => {
                if (e.data.type === "result" && e.data.symbol === this.$props.symbol) {
                    ok(`Got result for ${this.$props.symbol}.`);
                    window.worker.removeEventListener("message", handler);

                    info(`Updating chart - ${this.$props.symbol}...`);
                    this.chart.data.datasets.push({
                        data: e.data.result.map((line, index) => {return {x: index + this.data.length, y: line[1]}}),
                        xAxisId: "x-axis",
                        label: "Predicted",
                        borderColor: "rgb(24, 180, 128)",
                        backgroundColor: "rgba(24, 180, 128, 0.2)",
                        borderWidth: 2
                    });
                    this.chart.update();
                    ok(`Updated chart - ${this.$props.symbol}.`);
                }
            };
            window.worker.addEventListener("message", handler);
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