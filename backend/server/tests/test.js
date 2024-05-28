// start.js
const { spawn } = require('child_process');

function runPythonScript(inputString, callback) {
    const pythonProcess = spawn('python', ['test.py']);

    let output = '';
    pythonProcess.stdout.on('data', (data) => {
        output += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
        console.error(`Error: ${data.toString()}`);
    });

    pythonProcess.on('close', (code) => {
        if (code !== 0) { 
            return callback(`Python process exited with code ${code}`, null);
        }
        callback(null, output.trim());
    });

    pythonProcess.stdin.write(inputString + '\n');
    pythonProcess.stdin.end();
}

function callback(error, output){ 
    if (error) {
      console.error(error);
    } else {
      console.log(`Python output:\n${output}`);
    }
    console.log('Node.js script completed.');
}

function main() {
    const inputString = 'Hello from Node.js via start.js';
    console.log(`Sending to Python: ${inputString}`);
    
    runPythonScript(inputString, callback);
}

main();

