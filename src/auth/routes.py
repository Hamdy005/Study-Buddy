import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.database import get_auth_supabase
from src.store import create_user, get_user_by_email, delete_user_data, update_user_profile, get_user_by_id
from src.dependencies import get_current_user_id, get_current_user
from fastapi import Depends
from typing import Optional

router = APIRouter(prefix="/api/auth", tags=["Auth"])

@router.delete("/me")
async def delete_account(user_id: str = Depends(get_current_user_id)):
    delete_user_data(user_id)
    return {"status": "success", "message": "Account data deleted"}


class ProfileUpdateRequest(BaseModel):
    name: Optional[str] = None
    avatar_url: Optional[str] = None
    theme: Optional[str] = None


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
        or "@placeholder.ai" in email
        or "@studymate.ai" in email
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
                        res_upd = _table_supabase("profiles").insert(data).execute()
                        if res_upd.data:
                            user = _map_profile(res_upd.data[0])
                    except Exception:
                        # Profile row probably already exists — do a targeted update instead
                        try:
                            res_upd = (
                                _table_supabase("profiles")
                                .update(data)
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
        # Last resort: synthetic profile from the JWT claims so the UI doesn't break
        user_obj = current_user
        uid = getattr(user_obj, "id", None) or (user_obj.get("id") if isinstance(user_obj, dict) else None)
        if uid:
            from src.store import _map_profile
            meta = getattr(user_obj, "user_metadata", {}) or {}
            user = _map_profile({
                "id": uid,
                "display_name": meta.get("full_name") or meta.get("name") or "User",
                "email": getattr(user_obj, "email", "") or "",
                "avatar_url": "",
                "daily_requests": 0,
                "last_request_date": "",
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
