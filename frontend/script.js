document.addEventListener("DOMContentLoaded", function() {
    const chatWindow = document.getElementById('chat-window');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const debugInfo = document.getElementById('debug-info');

    // Fungsi untuk menambahkan pesan ke jendela chat
    function addMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', `${sender}-message`);
        const p = document.createElement('p');
        p.innerHTML = text.replace(/\n/g, '<br>'); // Mengganti newline dengan <br>
        messageDiv.appendChild(p);
        chatWindow.appendChild(messageDiv);
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }
    
    // Fungsi untuk menampilkan status 'mengetik...' dari bot
    function showBotTyping() {
        let typingMessage = document.querySelector('.bot-typing');
        if (!typingMessage) {
            typingMessage = document.createElement('div');
            typingMessage.classList.add('message', 'bot-message', 'bot-typing');
            typingMessage.innerHTML = `<p><span>.</span><span>.</span><span>.</span></p>`;
            chatWindow.appendChild(typingMessage);
            chatWindow.scrollTop = chatWindow.scrollHeight;
        }
    }

    // Fungsi untuk menghapus status 'mengetik...'
    function hideBotTyping() {
        const typingMessage = document.querySelector('.bot-typing');
        if (typingMessage) {
            chatWindow.removeChild(typingMessage);
        }
    }

    // Fungsi utama untuk mengirim pesan ke backend
    async function sendMessage() {
        const messageText = userInput.value.trim();
        if (messageText === '') return;

        addMessage(messageText, 'user');
        userInput.value = '';
        showBotTyping();

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ user_input: messageText }),
            });

            hideBotTyping();

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            
            // Tampilkan jawaban bot
            addMessage(data.answer, 'bot');
            
            // Tampilkan info debug di panel sebelah kanan
            debugInfo.textContent = data.debug_info;

        } catch (error) {
            hideBotTyping();
            console.error('Error:', error);
            addMessage('Maaf, terjadi kesalahan saat menghubungi server.', 'bot');
            debugInfo.textContent = `Error: ${error.message}`;
        }
    }

    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            sendMessage();
        }
    });
});