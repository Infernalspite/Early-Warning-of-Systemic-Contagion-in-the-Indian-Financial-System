import http.server
import socketserver
import os

PORT = 8000
DIRECTORY = "web"

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Serve from the 'web' directory
        super().__init__(*args, directory=DIRECTORY, **kwargs)

# Run the server
if __name__ == "__main__":
    # Ensure directory exists
    if not os.path.exists(DIRECTORY):
        os.makedirs(DIRECTORY, exist_ok=True)
        
    print("============================================================")
    print("SYSTEMIC CONTAGION ENGINE -- LOCAL WEB SERVER")
    print("============================================================")
    print(f"Server starting at http://localhost:{PORT}")
    print("Open your browser and navigate to the address above.")
    print("Press Ctrl+C to stop the server.")
    print("============================================================")


    # Allow address reuse to prevent "Address already in use" errors on quick restarts
    socketserver.TCPServer.allow_reuse_address = True
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping server...")
            httpd.server_close()
