const chatInput = document.querySelector(".chat-input textarea");
const sendChatBtn = document.querySelector(".chat-input span");
const chatbox = document.querySelector(".chatbox");
const chatbotToggler = document.querySelector(".chatbot-toggler");
const chatbotCloseBtn = document.querySelector(".close-btn");

let userMessage;
const inputInitHeight = chatInput.scrollHeight;

const createChatLi = (message, className) => {
  const chatLi = document.createElement("li");
  chatLi.classList.add("chat", className);
  let chatContent =
    className === "outgoing"
      ? `<p></p>`
      : `<span class="material-symbols-outlined"> smart_toy </span><p></p>`;
  chatLi.innerHTML = chatContent;
  chatLi.querySelector("p").textContent = message;
  return chatLi;
};

const generateResponse = (incomingChatLi, userMessage) => {
  const API_URL = "http://127.0.0.1:5000/ml_model"; // Change to your backend URL
  const messageElement = incomingChatLi.querySelector("p");

  const requestOptions = {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      message: userMessage,
    }),
  };

  fetch(API_URL, requestOptions)
    .then((res) => res.json())
    .then((data) => {
      messageElement.textContent = data.response;
    })
    .catch((error) => {
      messageElement.classList.add("error");
      messageElement.textContent =
        "Oops! Something went wrong. Please try again";
    })
    .finally(() => chatbox.scrollTo(0, chatbox.scrollHeight));
};

const handleChat = (userMessage) => {
  if (!userMessage) return;
  chatInput.value = "";

  chatbox.appendChild(createChatLi(userMessage, "outgoing"));
  chatbox.scrollTo(0, chatbox.scrollHeight);

  setTimeout(() => {
    const incomingChatLi = createChatLi("Thinking...", "incoming");
    chatbox.appendChild(incomingChatLi);
    chatbox.scrollTo(0, chatbox.scrollHeight);

    generateResponse(incomingChatLi, userMessage);
  }, 600);
};

// Function to handle initial greeting without triggering generateResponse immediately
const handleInitialGreeting = () => {
  const initialMessage =
    "Hi! Welcome to Workhub 24. How can I help you with that?";
  chatbox.appendChild(createChatLi(initialMessage, "incoming"));
  chatbox.scrollTo(0, chatbox.scrollHeight);
};

chatInput.addEventListener("input", () => {
  chatInput.style.height = `${inputInitHeight}px`;
  chatInput.style.height = `${chatInput.scrollHeight}px`;
});

sendChatBtn.addEventListener("click", () => handleChat(chatInput.value.trim()));

chatbotCloseBtn.addEventListener("click", () =>
  document.body.classList.remove("show-chatbot")
);
chatbotToggler.addEventListener("click", () =>
  document.body.classList.toggle("show-chatbot")
);

document.querySelector(".support-service").addEventListener("click", () => {
  const supportMessage = "Support Service";
  chatbox.appendChild(createChatLi(supportMessage, "outgoing"));
  chatbox.scrollTo(0, chatbox.scrollHeight);
  setTimeout(handleInitialGreeting, 600); // Call handleInitialGreeting after a delay
});

document.querySelector(".issue-ticket").addEventListener("click", () => {
  handleChat("Issue a Ticket");
  setTimeout(handleInitialGreeting, 600); // Also handle initial greeting after issuing a ticket
});
