document.addEventListener('DOMContentLoaded', (event) => {
    document.getElementById('square1').addEventListener('click', () => handleSquareClick(1));
    document.getElementById('square2').addEventListener('click', () => handleSquareClick(2));
    document.getElementById('square3').addEventListener('click', () => handleSquareClick(3));
    document.getElementById('square4').addEventListener('click', () => handleSquareClick(4));
    document.getElementById('square5').addEventListener('click', () => handleSquareClick(5));
    document.getElementById('square6').addEventListener('click', () => handleSquareClick(6));
    document.getElementById('square7').addEventListener('click', () => handleSquareClick(7));
    document.getElementById('square8').addEventListener('click', () => handleSquareClick(8));
    document.getElementById('square9').addEventListener('click', () => handleSquareClick(9));
});


let currentPlayer = 'X';
const board = Array(9).fill(null);

function handleSquareClick(index) {
    //toggleColor(document.getElementById(`square${index}`))
    postTile(index)
    console.log(board[index -1])
    if (board[index - 1] || checkWinner()) {
        return;
    }
    board[index - 1] = currentPlayer;

    let square;

    if(currentPlayer === 'O'){
    document.getElementById(`so${index}`).style.display = 'block';
    }
    else {
    document.getElementById(`sx${index}`).style.display = 'block';
  }

    if (checkWinner()) {
//        document.getElementById('status').textContent = `Player ${currentPlayer} wins!`;
    //} else if (board.every(square => square)) {
     //   document.getElementById('status').textContent = `It's a draw!`;
    } else {
        currentPlayer = currentPlayer === 'X' ? 'O' : 'X';
    }
}

function checkWinner() {
  const winningCombinations = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8], // rows
    [0, 3, 6], [1, 4, 7], [2, 5, 8], // columns
    [0, 4, 8], [2, 4, 6]             // diagonals
  ];

  for (const combination of winningCombinations) {
    const [a, b, c] = combination;
    if (board[a] && board[a] === board[b] && board[a] === board[c]) {
      console.log(`${currentPlayer} is the winner`)
      var squares = document.getElementsByClassName('square')
      for(var i = 0 ; i < squares.length; i++){
         squares[i].style.backgroundColor = 'blue' 
      }
      return true;
    }
  }
  return false;
}


function postTile(tile) {
    const data = { tile: tile };

    fetch('/tile', {
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
