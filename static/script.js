// Get elements from the DOM
const dropArea = document.getElementById('dropArea');
const imageUpload = document.getElementById('imageUpload');
const outputImage = document.getElementById('outputImage');
const submitButton = document.getElementById('submitButton');
const clearButton = document.getElementById('clearButton');
const uploadPreview = document.getElementById('uploadPreview');
const uploadText = document.getElementById('uploadText');

// Trigger file input when clicking on drop area
dropArea.addEventListener('click', () => imageUpload.click());

// Handle image preview after file selection
imageUpload.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            uploadPreview.src = e.target.result;
            uploadPreview.style.display = 'block';
            uploadText.style.display = 'none'; // Hide the text after image upload
        };
        reader.readAsDataURL(file);
    }
});

// Handle drag-and-drop functionality
dropArea.addEventListener('dragover', (event) => {
    event.preventDefault();
    dropArea.classList.add('active');
});

dropArea.addEventListener('dragleave', () => {
    dropArea.classList.remove('active');
});

dropArea.addEventListener('drop', (event) => {
    event.preventDefault();
    dropArea.classList.remove('active');
    const file = event.dataTransfer.files[0];
    imageUpload.files = event.dataTransfer.files; // Add the file to input
    const reader = new FileReader();
    reader.onload = (e) => {
        uploadPreview.src = e.target.result;
        uploadPreview.style.display = 'block';
        uploadText.style.display = 'none';
    };
    reader.readAsDataURL(file);
});

// Handle form submission
submitButton.addEventListener('click', () => {
    if (!imageUpload.files.length) {
        alert('Please upload an image first!');
        return;
    }

    // Create FormData object and append the image file
    const formData = new FormData();
    formData.append('image', imageUpload.files[0]);

    // Send the image to the server for processing
    fetch('/', {
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        // Update the output image with the colorized image
        outputImage.src = `static/uploads/${data.colorized_image}`;
    })
    .catch(error => console.error('Error:', error));
});

// Clear the image preview and file input
clearButton.addEventListener('click', () => {
    imageUpload.value = '';
    uploadPreview.style.display = 'none';
    uploadText.style.display = 'block'; // Show the text again when cleared
    outputImage.src = '';
});
