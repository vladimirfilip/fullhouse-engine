# Fullhouse Hackathon — Bot Sandbox Container
# One container per bot per match. Stateless, disposable.
#
# Security properties enforced at runtime (see match.py docker run call):
#   --network none          no internet access
#   --memory 768m           OOM-killed if exceeded (was 256m, bumped May 2026)
#   --cpus 0.5              half a core max
#   --read-only             filesystem is read-only
#   --no-new-privileges     can't escalate permissions
#   --user 1000:1000        non-root

FROM python:3.11-slim

# Install only what bots are allowed to use
# Participants can request additions via requirements.txt — we vet and rebuild
RUN pip install --no-cache-dir \
    eval7==0.1.7 \
    numpy==1.26.4 \
    scipy==1.13.0 \
    treys==0.1.8 \
    scikit-learn==1.5.2

# Create non-root user
RUN useradd -u 1000 -m botuser

# Runner lives at a fixed path — mounted or baked in
COPY runner.py /runner.py
RUN chmod +r /runner.py

# Bot code is mounted at runtime: -v /path/to/bot.py:/bot/bot.py:ro
# We do NOT copy bot code into the image — each bot gets its own volume mount

USER botuser
WORKDIR /home/botuser

# stdin/stdout is the communication channel
CMD ["python3", "-u", "/runner.py"]
