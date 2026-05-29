import uuid
from fastapi import APIRouter, HTTPException
from fastapi import Depends
from typing import Optional

from src.database import get_auth_supabase
from src.store import create_user, get_user_by_email, delete_user_data, update_user_profile, get_user_by_id
from src.dependencies import get_current_user_id, get_current_user
from .schemas import ProfileUpdateRequest

PLACEHOLDER_DOMAINS = ["@placeholder.ai", "@studymate.ai"]

router = APIRouter(prefix="/api/auth", tags=["Auth"])

@router.delete("/me")
async def delete_account(user_id: str = Depends(get_current_user_id)):
    delete_user_data(user_id)
    return {"status": "success", "message": "Account data deleted"}


@router.get("/profile")
async def get_profile(
    user_id: str = Depends(get_current_user_id),
    current_user=Depends(get_current_user)
):
    user = get_user_by_id(user_id)

    # Fast-path: profile row exists and has a real email — return immediately
    email = user.get("email", "") if user else ""
    is_placeholder = (
        not user
        or not email
        or any(domain in email for domain in PLACEHOLDER_DOMAINS)
    )

    if is_placeholder:
        # First-ever login: pull real data from Supabase Auth admin API and persist it
        from src.database import get_supabase
        supabase = get_supabase()
        if supabase:
            try:
                res = supabase.auth.admin.get_user_by_id(user_id)
                if res.user and res.user.email and "@" in res.user.email:
                    real_email = res.user.email
                    real_name = res.user.user_metadata.get("name")

                    from src.store import _table_supabase, _map_profile
                    data = {"id": user_id, "email": real_email}
                    if real_name:
                        data["display_name"] = real_name

                    try:
                        # Use upsert so we NEVER overwrite daily_requests / last_request_date
                        # on subsequent sign-ins.  Only id/email/display_name are safe to set.
                        client = supabase  # already resolved above
                        res_upd = (
                            client.table("profiles")
                            .upsert(data, on_conflict="id", ignore_duplicates=False)
                            .execute()
                        )
                        if res_upd.data:
                            user = _map_profile(res_upd.data[0])
                    except Exception:
                        # Upsert failed — fall back to a plain update (never resets usage)
                        try:
                            # Strip the id from the update payload to avoid PK conflicts
                            update_data = {k: v for k, v in data.items() if k != "id"}
                            res_upd = (
                                _table_supabase("profiles")
                                .update(update_data)
                                .eq("id", user_id)
                                .execute()
                            )
                            if res_upd.data:
                                user = _map_profile(res_upd.data[0])
                        except Exception:
                            pass
            except Exception:
                pass

    if not user:
        # Last resort: synthetic profile from the JWT claims so the UI doesn't break.
        # We still try to fetch real usage from the DB to avoid resetting the counter.
        user_obj = current_user
        uid = getattr(user_obj, "id", None) or (user_obj.get("id") if isinstance(user_obj, dict) else None)
        if uid:
            from src.store import _map_profile, get_usage
            meta = getattr(user_obj, "user_metadata", {}) or {}
            # Fetch real usage so the fallback profile doesn't reset the counter to 0
            real_usage = get_usage(uid)
            user = _map_profile({
                "id": uid,
                "display_name": meta.get("full_name") or meta.get("name") or "User",
                "email": getattr(user_obj, "email", "") or "",
                "avatar_url": "",
                "daily_requests": real_usage.get("used", 0),
                "last_request_date": __import__('datetime').date.today().isoformat(),
                "_is_fallback": True,
            })

    if not user:
        raise HTTPException(404, "Profile not found")
    return {"status": "success", "user": user}


@router.patch("/profile")
async def update_profile(body: ProfileUpdateRequest, user_id: str = Depends(get_current_user_id)):
    try:
        updated_user = update_user_profile(
            user_id,
            name=body.name,
            avatar_url=body.avatar_url,
            theme=body.theme
        )
        return {"status": "success", "user": updated_user}
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Failed to update profile: {e}")
