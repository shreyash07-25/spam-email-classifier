const API_URL = "https://spam-email-classifier-tq7u.onrender.com";

const messageInput = document.getElementById("message");
const charCount = document.getElementById("charCount");

const checkButton = document.getElementById("checkButton");
const buttonContent = document.getElementById("buttonContent");
const loadingContent = document.getElementById("loadingContent");

const result = document.getElementById("result");
const prediction = document.getElementById("prediction");
const resultDescription = document.getElementById("resultDescription");

const resultIcon = document.getElementById("resultIcon");
const confidenceValue = document.getElementById("confidenceValue");
const confidenceBar = document.getElementById("confidenceBar");


/* Character counter */

messageInput.addEventListener("input", () => {

    charCount.textContent =
        `${messageInput.value.length} / 5000`;

});


/* Analyze */

async function checkMessage() {

    const message = messageInput.value.trim();

    if (!message) {
        alert("Please enter a message first.");
        return;
    }

    setLoading(true);

    try {

        const response = await fetch(`${API_URL}/predict`, {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                message: message
            })

        });


        const data = await response.json();


        if (!response.ok) {
            throw new Error(data.detail || "API request failed.");
        }


        if (data.error) {
            throw new Error(data.error);
        }


        showResult(data);


    } catch (error) {

        console.error(error);

        alert(
            "Unable to connect to the API. " +
            "Make sure FastAPI is running."
        );

    } finally {

        setLoading(false);

    }
}


/* Show result */

function showResult(data) {

    const confidence =
        data.confidence * 100;


    result.classList.remove("hidden", "spam");


    prediction.textContent =
        data.prediction;


    confidenceValue.textContent =
        `${confidence.toFixed(2)}%`;


    confidenceBar.style.width =
        `${confidence}%`;


    if (data.prediction === "SPAM") {

        result.classList.add("spam");

        resultIcon.textContent = "!";

        resultDescription.textContent =
            "This message has characteristics commonly associated with spam.";

    } else {

        resultIcon.textContent = "✓";

        resultDescription.textContent =
            "This message appears to be a legitimate message.";

    }

}


/* Loading state */

function setLoading(loading) {

    checkButton.disabled = loading;


    if (loading) {

        buttonContent.classList.add("hidden");
        loadingContent.classList.remove("hidden");

    } else {

        buttonContent.classList.remove("hidden");
        loadingContent.classList.add("hidden");

    }

}


/* Clear */

function clearMessage() {

    messageInput.value = "";

    charCount.textContent = "0 / 5000";

    result.classList.add("hidden");

    confidenceBar.style.width = "0%";

}


/* Example messages */

function useExample(type) {

    if (type === "normal") {

        messageInput.value =
            "Hey, are you free tomorrow? Let's meet at the library after class.";

    } else {

        messageInput.value =
            "Congratulations! You have been selected to receive a FREE cash prize. Click now to claim your reward!";

    }


    charCount.textContent =
        `${messageInput.value.length} / 5000`;


    messageInput.focus();

}


