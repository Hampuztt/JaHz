const express = require('express');
const app = express();
const port = 3000;

const { spawn } = require('child_process');
const pythonProcess = spawn('python', ['tictac.py']);
let message = 'no message'

app.get('/', (req, res) => {
    res.send(message);
});

app.listen(port, () => {
    console.log(`Example app listening at http://localhost:${port}`); 
    pythonProcess.stdout.on('data', (data) => {
    message = `${data}` //will give weird values in future
    console.log(`stdout: ${data}`);
    });
});

app.post('/tile', (req, res) => {
    const tile = req.body.tile;
    pythonProcess.stdin.write(tile); 
    pythonProcess.stdin.end();
});
