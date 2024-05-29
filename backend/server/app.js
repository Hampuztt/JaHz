require('dotenv').config()
const express = require('express');
const cors = require('cors')
const path = require('path')
const app = express();
const port = process.env.PORT;

const { spawn } = require('child_process');
const pythonProcess = spawn('python3', ['tictac.py']);
let message = 'no message';

console.log('hello')

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, '../..', 'frontend')))

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, '../..', 'frontend', 'index.html'))
});

app.listen(port, () => {
    console.log(`listening at port: "${port}`); 
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

