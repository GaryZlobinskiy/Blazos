this.window = this;

importScripts("//cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.0.0");

function info(message) {
    console.info(`[worker %cinfo%c] ${message}`, "color: cyan; font-weight: bold", "color: inherit; font-weight: initial");
}

function ok(message) {
    console.info(`[worker %cok%c] ${message}`, "color: #0f0; font-weight: bold", "color: inherit; font-weight: initial");
}

ok("Spawned worker.");

let model;
let tfTensor;

onmessage = e => {
    info(`Received messsage of type ${e.data.type}`);
    if (e.data.type === "model") {
        model = e.data.model;
        ok("Loaded model.");
    } else if (e.data.type === "predict") {
        info(`Predicting for ${e.data.symbol} over ${e.data.days} days`);

        let data = e.data.data;

        console.log(tf.tensor([2, 3]));

        for (let i = 0; i < e.data.days; i++) {
            info(`Running neural network - ${e.data.symbol}...`);
            data.push([Math.random(), Math.random(), Math.random(), Math.random(), Math.random()]);
            // model.predict(new tf.tensor());
        }

        postMessage({type: "result", symbol: e.data.symbol, result: data.slice(60)});
    }
};