const express = require('express');
const cors = require('cors')
const app = express();
const port = 3000;

const { spawn } = require('child_process');
const pythonProcess = spawn('python3', ['tictac.py']);
let message = 'no message';

app.use(cors());
app.use(express.json());

app.get('/', (req, res) => {
    res.send(message);
});

app.listen(port, () => {
    console.log(`Example app listening at http://localhost:${port}`); 
    pythonProcess.stdout.on('data', (data) => {
        message = `${data}`; // will give weird values in future
        console.log(`stdout: ${data}`);
    });
})

app.get('/t', (req, res) => {
    res.send('how did you find mr.t?');
});

app.post('/tile', (req, res) => {
    const tile = req.body.tile;
    pythonProcess.stdin.write(tile + '\n');
    res.status(200).json({message: 'tile data recieved', data : req.body})
});

