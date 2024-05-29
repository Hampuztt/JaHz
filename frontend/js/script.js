document.addEventListener('DOMContentLoaded', (event) => {
    document.getElementById('square1').addEventListener('click', square1Click);
    document.getElementById('square2').addEventListener('click', square2Click);
    document.getElementById('square3').addEventListener('click', square3Click);
    document.getElementById('square4').addEventListener('click', square4Click);
    document.getElementById('square5').addEventListener('click', square5Click);
    document.getElementById('square6').addEventListener('click', square6Click);
    document.getElementById('square7').addEventListener('click', square7Click);
    document.getElementById('square8').addEventListener('click', square8Click);
    document.getElementById('square9').addEventListener('click', square9Click);
});

function postTile(tile) {
    const data = { tile: tile };

    fetch('http://localhost:3000/tile', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        console.log('Success:', data);
    })
    .catch((error) => {
        console.error('Error:', error);
    });
}

function toggleColor(square) {
    if (square.style.backgroundColor === 'red') {
        square.style.backgroundColor = '#90EE90';
    } else {
        square.style.backgroundColor = 'red';
    }
}

function square1Click() {
    toggleColor(document.getElementById('square1'));
    postTile('a1')
}

function square2Click() {
    toggleColor(document.getElementById('square2'));
    postTile('a2')
}

function square3Click() {
    toggleColor(document.getElementById('square3'));
    postTile('a3')
}

function square4Click() {
    toggleColor(document.getElementById('square4'));
    postTile('b1')
}

function square5Click() {
    toggleColor(document.getElementById('square5'));
    postTile('b2')
}

function square6Click() {
    toggleColor(document.getElementById('square6'));
    postTile('b3')
}

function square7Click() {
    toggleColor(document.getElementById('square7'));
    postTile('c1')
}

function square8Click() {
    toggleColor(document.getElementById('square8'));
    postTile('c2')
}

function square9Click() {
    toggleColor(document.getElementById('square9'));
    postTile('c3')
}

